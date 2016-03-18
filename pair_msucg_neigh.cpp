/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Paul Crozier (SNL)
------------------------------------------------------------------------- */

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pair_msucg_neigh.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "domain.h"
#include "modify.h"
#include "fix.h"

#include <algorithm>

using namespace LAMMPS_NS;
using namespace MathConst;

#define MAXLINE 1024 // Used for state definition input.

/* ---------------------------------------------------------------------- */

PairMSUCG_NEIGH::PairMSUCG_NEIGH(LAMMPS *lmp) : Pair(lmp)
{
  respa_enable = 1;
  writedata = 1;
  countiter = 0;
  countneigh = 0;
  nmax = 0;

  substate_probability = NULL;
  substate_probability_partial = NULL;
  substate_probability_force = NULL;
  substate_cv_backforce = NULL;
  state_params_allocated = 0;

  comm_reverse = 3;
  comm_forward = 3;
}

/* ---------------------------------------------------------------------- */

PairMSUCG_NEIGH::~PairMSUCG_NEIGH()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);
  }
  if (state_params_allocated) {
    memory->destroy(n_states_per_type);
    memory->destroy(actual_types_from_state);
    memory->destroy(use_state_entropy);
    memory->destroy(chemical_potentials);
    memory->destroy(cv_thresholds);
    memory->destroy(threshold_radii);
  }
}

/* ---------------------------------------------------------------------- */

void PairMSUCG_NEIGH::threshold_prob_and_partial_from_cv(int type, double cv, double &prob, double &partial) {
  double tanh_factor = tanh((cv - cv_thresholds[type]) / (0.1 * cv_thresholds[type]));
  prob = 0.5 + 0.5 * tanh_factor;
  partial = 0.5 * (1.0 - tanh_factor * tanh_factor) / (0.1 * cv_thresholds[type]);
}

/* ---------------------------------------------------------------------- */

int PairMSUCG_NEIGH::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = substate_cv_backforce[i][0];
    buf[m++] = substate_cv_backforce[i][1];
    buf[m++] = substate_cv_backforce[i][2];
  }
  return m;
}

void PairMSUCG_NEIGH::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    substate_cv_backforce[j][0] += buf[m++];
    substate_cv_backforce[j][1] += buf[m++];
    substate_cv_backforce[j][2] += buf[m++];
  }
}

int PairMSUCG_NEIGH::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int i,j,m,jsubstate;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    for (jsubstate = 0; jsubstate < max_states_per_type - 1; jsubstate++) {
      buf[m++] = substate_probability[j][jsubstate];
      buf[m++] = substate_probability_partial[j][jsubstate];
      buf[m++] = substate_probability_force[j][jsubstate];
    }
  }
  return m;
}

void PairMSUCG_NEIGH::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last,isubstate;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    for (isubstate = 0; isubstate < max_states_per_type - 1; isubstate++) {
      substate_probability[i][isubstate] = buf[m++];
      substate_probability_partial[i][isubstate] = buf[m++];
      substate_probability_force[i][isubstate] = buf[m++];
    }
  }
}

/* ---------------------------------------------------------------------- */

double PairMSUCG_NEIGH::compute_proximity_function(int type, double distance) {
  double tanh_factor = tanh((distance - threshold_radii[type]) / (0.1 * threshold_radii[type]));
  return 0.5 * (1.0 - tanh_factor);
}

double PairMSUCG_NEIGH::compute_proximity_function_der(int type, double distance) {
  double tanh_factor = tanh((distance - threshold_radii[type]) / (0.1 * threshold_radii[type]));
  return 0.5 * (1.0 - tanh_factor * tanh_factor) / (0.1 * threshold_radii[type]);
  // Also changed the sign - to + because it will be considered in the substate_probability_force
}

/* ---------------------------------------------------------------------- */

void PairMSUCG_NEIGH::compute(int eflag, int vflag)
{
	int i,j,ii,jj,inum,jnum,itype,jtype;
  int itype_actual, jtype_actual;
	double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
	double rsq,r2inv,r6inv,forcelj,factor_lj;
  double distance,inumber_density,cv_force;
  int isubstate,jsubstate,ksubstate,alpha,beta;
  double alphaprob,betaprob;
  double i_prob_accounted, j_prob_accounted;
	/* Additional parameter */
	int *ilist,*jlist,*numneigh,**firstneigh;

  double pair_force; // For updating in the ev_tally routine
  double energy_lj; // Energy routine for ev_tally routine
  /*  = (double *)malloc(sizeof(double)*3) */
  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag, vflag);
  else evflag = vflag_fdotr = 0;

	double **x = atom->x;
	double **f = atom->f;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	double *special_lj = force->special_lj;
	int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  int nall = nlocal + atom->nghost;
	if (nall > nmax) {
  	nmax = nall;
    memory->grow(substate_probability, nall, max_states_per_type - 1, "pair/msucg:substate_probability");
    memory->grow(substate_probability_partial, nall, max_states_per_type - 1, "pair/msucg:substate_probability_partial");
    memory->grow(substate_probability_force, nall, max_states_per_type - 1, "pair/msucg:substate_probability_force");
    memory->grow(substate_cv_backforce, nall, 3, "pair/msucg:substate_cv_backforce");
  }

  for(i = 0; i < nall; i++) {
    for (j = 0; j < max_states_per_type - 1; j++) {
      substate_probability[i][j] = 0.0;
      substate_probability_partial[i][j] = 0.0;
      substate_probability_force[i][j] = 0.0;
    }
    substate_cv_backforce[i][0] = 0.0;
    substate_cv_backforce[i][1] = 0.0;
    substate_cv_backforce[i][2] = 0.0;
 	}

  // First loop: calculate the state probabilities.
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    itype_actual = actual_types_from_state[itype];
    inumber_density = 0.0;
    jlist = firstneigh[i];
    jnum = numneigh[i];
    
    if (n_states_per_type[itype_actual] > 1) {
      // For each particle with states, calculate the CV that 
      // controls the state probability, in this case density.
      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx * delx + dely * dely + delz * delz;
        jtype = type[j];
  
        if (rsq < cutsq[itype][jtype]) {
          distance = sqrt(rsq);
          inumber_density += compute_proximity_function(itype_actual, distance);
        }
      }
  
      // Keep track of the probability and its partial derivative.
      threshold_prob_and_partial_from_cv(itype_actual, inumber_density, substate_probability[i][0], substate_probability_partial[i][0]);
    } else {
      // For types without substates, simply assign p0 = 1.
      // (No partial derivatives.)
      substate_probability[i][0] = 1.0;
    }
    // printf("Particle %d has number_density %g and substate_probability %g given substate_threshold of %g for type %d with sigma cutoff : %g \n", i, inumber_density, substate_probability[i], cv_thresholds[itype_actual], itype, threshold_radii[itype_actual]);
  }
  // Communicate state probabilities forward.
  comm->forward_comm_pair(this);

  // Second loop: calculate all forces that do not depend on 
  // probability derivatives and against the state distribution 
  // as well.
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    itype_actual = actual_types_from_state[itype];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    // Compute single-body forces conjugate to state change.
    // Apply to each of the state probabilities except the last,
    // which is always kept implicit.
    if (n_states_per_type[itype_actual] > 1) {
      i_prob_accounted = 0.0;
      // For each substate but the last, calculate the derivative
      // of the free energy with respect to probability.
      for (isubstate = 0; isubstate < n_states_per_type[itype_actual] - 1; isubstate++) {
        // Calculate one-body-state entropic forces.
        if (use_state_entropy[itype_actual]) {
          substate_probability_force[i][isubstate] -= kT * log(substate_probability[i][isubstate]);
        }
        // Calculate one-body-state potential forces.
        substate_probability_force[i][isubstate] -= chemical_potentials[itype + isubstate];
        i_prob_accounted += substate_probability[i][isubstate];
      }
      // For the last substate, use conservation of probability to write
      // its effect as force mediated through the other probabilities.
      if (use_state_entropy[itype_actual]) {
        for (isubstate = 0; isubstate < n_states_per_type[itype_actual] - 1; isubstate++) {
          substate_probability_force[i][isubstate] += kT * log(1 - i_prob_accounted);
        }
      }
    }

    // Compute two-body forces at fixed state and effects of the
    // two body potential on state change.
    for (jj = 0; jj < jnum; jj++) {
      energy_lj = 0.0;
      pair_force = 0.0;
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];
      jtype_actual = actual_types_from_state[jtype];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0 / rsq;
        r6inv = r2inv * r2inv * r2inv;
        
        // Loop over all possible substates of particle i.
        i_prob_accounted = 0;
        for (isubstate = 0; isubstate < n_states_per_type[itype_actual]; isubstate++) {
          alpha = itype + isubstate;
          if (n_states_per_type[itype_actual] > 1) {
            if (isubstate < n_states_per_type[itype_actual] - 1) {
              alphaprob = substate_probability[i][isubstate];
              i_prob_accounted += substate_probability[i][isubstate];
            } else {
              alphaprob = (1 - i_prob_accounted);
            }
          } else {
            alphaprob = 1.0;
          }
          
          // Iterate over all possible substates of particle j.
          j_prob_accounted = 0;
          for (jsubstate = 0; jsubstate < n_states_per_type[jtype_actual]; jsubstate++) {
            beta = jtype + jsubstate;
            if (n_states_per_type[jtype_actual] > 1) {
              if (jsubstate < n_states_per_type[jtype_actual] - 1) {
                betaprob = substate_probability[j][jsubstate];
                j_prob_accounted += substate_probability[j][jsubstate];
              } else {
                betaprob = (1 - j_prob_accounted);
              }
            } else {
              betaprob = 1.0;
            }
            // fprintf(screen, "alpha %d (%d): %g beta %d (%d): %g \n", alpha, i,alphaprob, beta, j,betaprob);
            
            // Calculate the usual force between the particles.
            forcelj = r6inv * (lj1[alpha][beta] * r6inv - lj2[alpha][beta]);
            // Scale the usual pair force by current state weights.
            fpair = factor_lj * forcelj * r2inv * alphaprob * betaprob;
            // Accumulate.
            f[i][0] += fpair * delx;
            f[i][1] += fpair * dely;
            f[i][2] += fpair * delz;
            pair_force += fpair;
            
            // Calculate the usual pair energy.
            evdwl = r6inv*(lj3[alpha][beta]*r6inv-lj4[alpha][beta]) - offset[alpha][beta];
            evdwl *= factor_lj;
            // Scale the usual pair force by current state weights.
            energy_lj += evdwl * alphaprob * betaprob;
            
            // Apply the state-specific pair energy as a conjugate
            // to the state distribution.
            if (n_states_per_type[itype_actual] > 1) {
              if (isubstate < n_states_per_type[itype_actual] - 1) {
                substate_probability_force[i][isubstate] -= betaprob * evdwl;
              } else {
                for (ksubstate = 0; ksubstate < n_states_per_type[itype_actual] - 1; ksubstate++) {
                  substate_probability_force[i][ksubstate] += betaprob * evdwl;
                }
              }
            }
          }
        }
        if (evflag) ev_tally(i,j,nlocal,newton_pair,energy_lj,0.0,pair_force,delx,dely,delz); // ev_tally has to be inside of the j loop? 
      }
    }
  }
  
  // Third loop: calculate forces from probability derivatives 
  // on local atoms.
  // Forces from local atom probabilities on ghosts must be 
  // reverse communicated.
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    itype_actual = actual_types_from_state[itype];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    if (n_states_per_type[itype_actual] > 1) {

      for (isubstate = 0; isubstate < n_states_per_type[itype_actual] - 1; isubstate++) {
        
        // Convert force against the state to force against the CV
        // by using the partial of state with respect to CV.
        cv_force = substate_probability_force[i][isubstate] * substate_probability_partial[i][isubstate];
  
        // Apply the force against the CV, in this case density.
        for (jj = 0; jj < jnum; jj++) {
          j = jlist[jj];
          delx = xtmp - x[j][0];
          dely = ytmp - x[j][1];
          delz = ztmp - x[j][2];
          rsq = delx * delx + dely * dely + delz * delz;
          jtype = type[j];
  
          // Distribute the force down to every pair of particles
          // contributing to the density.
          if (rsq < cutsq[itype][jtype]) {
            distance = sqrt(rsq);
            fpair = cv_force * compute_proximity_function_der(itype_actual, distance) / distance;
            substate_cv_backforce[i][0] += fpair * delx;
            substate_cv_backforce[i][1] += fpair * dely;
            substate_cv_backforce[i][2] += fpair * delz;
            substate_cv_backforce[j][0] -= fpair * delx;
            substate_cv_backforce[j][1] -= fpair * dely;
            substate_cv_backforce[j][2] -= fpair * delz;
          }
        }
      }
    }
  }
  comm->reverse_comm_pair(this);
  
  // Add the CV forces to the other forces.
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    f[i][0] += substate_cv_backforce[i][0];
    f[i][1] += substate_cv_backforce[i][1];
    f[i][2] += substate_cv_backforce[i][2];
  }

  //for (ii = 0; ii < inum; ii++) {
  //  i = ilist[ii];
  //  fprintf(screen, "Force: (%8.4E, %8.4E, %8.4E) on %d(%8.4E, %8.4E, %8.4E)  \n", f[i][0], f[i][1], f[i][2], i,x[i][0],x[i][1],x[i][2]);
  //}

	if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairMSUCG_NEIGH::compute_inner()
{
  error->all(FLERR, "Work not done here!"); /*---YP--- Haven't done this part yet */
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,fpair;
  double rsq,r2inv,r6inv,forcelj,factor_lj,rsw;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = listinner->inum;
  ilist = listinner->ilist;
  numneigh = listinner->numneigh;
  firstneigh = listinner->firstneigh;

  double cut_out_on = cut_respa[0];
  double cut_out_off = cut_respa[1];

  double cut_out_diff = cut_out_off - cut_out_on;
  double cut_out_on_sq = cut_out_on*cut_out_on;
  double cut_out_off_sq = cut_out_off*cut_out_off;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cut_out_off_sq) {
        r2inv = 1.0/rsq;
        r6inv = r2inv*r2inv*r2inv;
        jtype = type[j];
        forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
        fpair = factor_lj*forcelj*r2inv;
        if (rsq > cut_out_on_sq) {
          rsw = (sqrt(rsq) - cut_out_on)/cut_out_diff;
          fpair *= 1.0 - rsw*rsw*(3.0 - 2.0*rsw);
        }

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairMSUCG_NEIGH::compute_middle()
{
  error->all(FLERR, "Work not done here!"); /*---YP--- Haven't done this part yet */

  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,fpair;
  double rsq,r2inv,r6inv,forcelj,factor_lj,rsw;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = listmiddle->inum;
  ilist = listmiddle->ilist;
  numneigh = listmiddle->numneigh;
  firstneigh = listmiddle->firstneigh;

  double cut_in_off = cut_respa[0];
  double cut_in_on = cut_respa[1];
  double cut_out_on = cut_respa[2];
  double cut_out_off = cut_respa[3];

  double cut_in_diff = cut_in_on - cut_in_off;
  double cut_out_diff = cut_out_off - cut_out_on;
  double cut_in_off_sq = cut_in_off*cut_in_off;
  double cut_in_on_sq = cut_in_on*cut_in_on;
  double cut_out_on_sq = cut_out_on*cut_out_on;
  double cut_out_off_sq = cut_out_off*cut_out_off;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cut_out_off_sq && rsq > cut_in_off_sq) {
        r2inv = 1.0/rsq;
        r6inv = r2inv*r2inv*r2inv;
        jtype = type[j];
        forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
        fpair = factor_lj*forcelj*r2inv;
        if (rsq < cut_in_on_sq) {
          rsw = (sqrt(rsq) - cut_in_off)/cut_in_diff;
          fpair *= rsw*rsw*(3.0 - 2.0*rsw);
        }
        if (rsq > cut_out_on_sq) {
          rsw = (sqrt(rsq) - cut_out_on)/cut_out_diff;
          fpair *= 1.0 + rsw*rsw*(2.0*rsw - 3.0);
        }

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairMSUCG_NEIGH::compute_outer(int eflag, int vflag)
{
  error->all(FLERR, "Work not done here!"); /*---YP--- Haven't done this part yet */

  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r2inv,r6inv,forcelj,factor_lj,rsw;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = listouter->inum;
  ilist = listouter->ilist;
  numneigh = listouter->numneigh;
  firstneigh = listouter->firstneigh;

  double cut_in_off = cut_respa[2];
  double cut_in_on = cut_respa[3];

  double cut_in_diff = cut_in_on - cut_in_off;
  double cut_in_off_sq = cut_in_off*cut_in_off;
  double cut_in_on_sq = cut_in_on*cut_in_on;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        if (rsq > cut_in_off_sq) {
          r2inv = 1.0/rsq;
          r6inv = r2inv*r2inv*r2inv;
          forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
          fpair = factor_lj*forcelj*r2inv;
          if (rsq < cut_in_on_sq) {
            rsw = (sqrt(rsq) - cut_in_off)/cut_in_diff;
            fpair *= rsw*rsw*(3.0 - 2.0*rsw);
          }

          f[i][0] += delx*fpair;
          f[i][1] += dely*fpair;
          f[i][2] += delz*fpair;
          if (newton_pair || j < nlocal) {
            f[j][0] -= delx*fpair;
            f[j][1] -= dely*fpair;
            f[j][2] -= delz*fpair;
          }
        }

        if (eflag) {
          r2inv = 1.0/rsq;
          r6inv = r2inv*r2inv*r2inv;
          evdwl = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) -
            offset[itype][jtype];
          evdwl *= factor_lj;
        }

        if (vflag) {
          if (rsq <= cut_in_off_sq) {
            r2inv = 1.0/rsq;
            r6inv = r2inv*r2inv*r2inv;
            forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
            fpair = factor_lj*forcelj*r2inv;
          } else if (rsq < cut_in_on_sq)
            fpair = factor_lj*forcelj*r2inv;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairMSUCG_NEIGH::allocate()
{
  allocated = 1;
  int n = atom->ntypes;
  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
  memory->create(offset,n+1,n+1,"pair:offset");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairMSUCG_NEIGH::settings(int narg, char **arg)
{
  if (narg != 2) error->all(FLERR,"Illegal pair_style command");

  cut_global = force->numeric(FLERR,arg[0]);

  // Read in a state definition file
  read_state_settings(arg[1]);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i+1; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

void PairMSUCG_NEIGH::read_state_settings(const char *file) {
  char *eof;
  char line[MAXLINE];
  char state_type[MAXLINE];
  char entropy_spec[MAXLINE];

  // Open the state settings file.
  FILE* fp = fopen(file, "r");
  if (fp == NULL) {
    char str[128];
    sprintf(str, "Cannot open file %s", file);
    error->one(FLERR, str);
  }

  // Read the total number of actual types and total number of states.
  eof = fgets(line, MAXLINE, fp);
  if (eof == NULL) error->one(FLERR,"Unexpected end of MSUCG state settings file");
  sscanf(line,"%d %d", &n_actual_types, &n_total_states);

  // Allocate space for storing state settings based on the number
  // of actual types.
  memory->create(n_states_per_type, n_actual_types + 1, "pair:n_states_per_type");
  memory->create(actual_types_from_state, n_total_states + 1, "pair:n_states_per_type");
  memory->create(use_state_entropy, n_actual_types + 1, "pair:n_states_per_type");
  memory->create(chemical_potentials, n_total_states + 1, "pair:n_states_per_type");
  memory->create(cv_thresholds, n_actual_types + 1, "pair:n_states_per_type");
  memory->create(threshold_radii, n_actual_types + 1, "pair:n_states_per_type");
  
  state_params_allocated = 1;

  for (int i = 0; i <= n_total_states; i++) {
    chemical_potentials[i] = 0.0;
    actual_types_from_state[i] = 0;
  }
  for (int i = 0; i <= n_actual_types; i++) {
    n_states_per_type[i] = 0;
    use_state_entropy[i] = 0;
    cv_thresholds[i] = 0.0;
    threshold_radii[i] = 0.0;
  }

  // For each actual type, read the number of states for that type, and
  // (if more than one) the density threshold, the threshold radius for
  // the density, and the one-body chemical potential for the state.
  int curr_state = 1;
  max_states_per_type = 2;
  for (int i = 1; i <= n_actual_types; i++) {
    // Read the number of states and way that they are assigned.
    eof = fgets(line, MAXLINE, fp);
    if (eof == NULL) error->one(FLERR,"Unexpected end of MSUCG state settings file");
    sscanf(line, "%d %s %s", &n_states_per_type[i], state_type, entropy_spec);
    max_states_per_type = std::max(max_states_per_type, n_states_per_type[i]);
    if (strcmp(state_type, "use_entropy") == 0) {
      use_state_entropy[i] = 1;
    } else if (strcmp(state_type, "no_entropy") == 0) {
      use_state_entropy[i] = 0;
    }

    // If this type has more than one state, read further state parameters.
    if (n_states_per_type[i] > 1) {
      // Read state probability assignment parameters.
      if (strcmp(state_type, "density") == 0) {
        eof = fgets(line, MAXLINE, fp);
        if (eof == NULL) error->one(FLERR,"Unexpected end of MSUCG state settings file");
        sscanf(line, "%lg %lg", &cv_thresholds[i], &threshold_radii[i]);
      } else {
        error->one(FLERR,"Unknown state assignment type for MSUCG");
      }
      // Read state chemical potentials.
      eof = fgets(line, MAXLINE, fp);
      if (eof == NULL) error->one(FLERR,"Unexpected end of MSUCG state settings file");
      char *p = strtok(line, " ");
      for (int j = 0; j < n_states_per_type[i] - 1; j++) {
        sscanf(p, "%lg", &chemical_potentials[i + j]);
        p = strtok(NULL, " ");
      }
    }

    // Keep an up-to-date back-map from state ids to actual type ids.
    for (int j = 0; j < n_states_per_type[i]; j++) {
      actual_types_from_state[curr_state] = i;
      curr_state++;
    }
  }
  comm_forward = 3 * (max_states_per_type - 1);

  // Close after finishing.
  fclose(fp);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairMSUCG_NEIGH::coeff(int narg, char **arg)
{
  if (narg < 5 || narg > 6)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(arg[0],atom->ntypes,ilo,ihi);
  force->bounds(arg[1],atom->ntypes,jlo,jhi);

  double epsilon_one = force->numeric(FLERR,arg[2]);
  double sigma_one = force->numeric(FLERR,arg[3]);

  double cut_one = cut_global;

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairMSUCG_NEIGH::init_style()
{
  // request regular or rRESPA neighbor lists

  int irequest;
  int newton_pair = force->newton_pair;

  if (update->whichflag == 1 && strstr(update->integrate_style,"respa")) {
    int respa = 0;
    if (((Respa *) update->integrate)->level_inner >= 0) respa = 1;
    if (((Respa *) update->integrate)->level_middle >= 0) respa = 2;

    if (respa == 0) irequest = neighbor->request(this,instance_me);
    else if (respa == 1) {
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 1;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respainner = 1;
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 3;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respaouter = 1;
    } else {
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 1;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respainner = 1;
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 2;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respamiddle = 1;
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 3;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respaouter = 1;
    }

  } else{
    irequest = neighbor->request(this,instance_me);
    neighbor->requests[irequest]->half=0;
    neighbor->requests[irequest]->full=1;
  }
  // set rRESPA cutoffs

  if (strstr(update->integrate_style,"respa") &&
      ((Respa *) update->integrate)->level_inner >= 0)
    cut_respa = ((Respa *) update->integrate)->cutoff;
  else cut_respa = NULL;

  /*---YP--- Obtain thermostat temperature */

  double *pT = NULL;
  int pdim;

  for(int ifix = 0; ifix < modify->nfix; ifix++)
  {
    pT = (double*) modify->fix[ifix]->extract("t_target", pdim);
    if(pT) { T = (*pT); break; }
  }

  if(pT==NULL) error->all(FLERR, "Cannot locate temperature target from thermostat.");
  else fprintf(screen, "Pair/MSUCG Ensemble temperature is %lf.\n", T);

  kT = force->boltz * T;

  if(newton_pair != 0) error->all(FLERR, "Newton pair is turned on. It has to be turned off in local density UCG simulation.");

}

/* ----------------------------------------------------------------------
   neighbor callback to inform pair style of neighbor list to use
   regular or rRESPA
------------------------------------------------------------------------- */

void PairMSUCG_NEIGH::init_list(int id, NeighList *ptr)
{
  if (id == 0) list = ptr;
  else if (id == 1) listinner = ptr;
  else if (id == 2) listmiddle = ptr;
  else if (id == 3) listouter = ptr;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairMSUCG_NEIGH::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
                               sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
    /*
    double check = ((i-1)+(j-2)) * ((i-3)+(j-4));
    if (check != 0){
    	epsilon[i][j] = epsilon[i][j] * 0.20;
    }
    */
  }

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);

  if (offset_flag) {
    double ratio = sigma[i][j] / cut[i][j];
    offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio,12.0) - pow(ratio,6.0));
  } else offset[i][j] = 0.0;

  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  offset[j][i] = offset[i][j];

  // check interior rRESPA cutoff

  if (cut_respa && cut[i][j] < cut_respa[3])
    error->all(FLERR,"Pair cutoff < Respa interior cutoff");

  // compute I,J contribution to long-range tail correction
  // count total # of atoms of type I and J via Allreduce

  if (tail_flag) {
    int *type = atom->type;
    int nlocal = atom->nlocal;

    double count[2],all[2];
    count[0] = count[1] = 0.0;
    for (int k = 0; k < nlocal; k++) {
      if (type[k] == i) count[0] += 1.0;
      if (type[k] == j) count[1] += 1.0;
    }
    MPI_Allreduce(count,all,2,MPI_DOUBLE,MPI_SUM,world);

    double sig2 = sigma[i][j]*sigma[i][j];
    double sig6 = sig2*sig2*sig2;
    double rc3 = cut[i][j]*cut[i][j]*cut[i][j];
    double rc6 = rc3*rc3;
    double rc9 = rc3*rc6;
    etail_ij = 8.0*MY_PI*all[0]*all[1]*epsilon[i][j] *
      sig6 * (sig6 - 3.0*rc6) / (9.0*rc9);
    ptail_ij = 16.0*MY_PI*all[0]*all[1]*epsilon[i][j] *
      sig6 * (2.0*sig6 - 3.0*rc6) / (9.0*rc9);
  }

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairMSUCG_NEIGH::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairMSUCG_NEIGH::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&epsilon[i][j],sizeof(double),1,fp);
          fread(&sigma[i][j],sizeof(double),1,fp);
          fread(&cut[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairMSUCG_NEIGH::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
  fwrite(&tail_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairMSUCG_NEIGH::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    fread(&cut_global,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
    fread(&tail_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
  MPI_Bcast(&tail_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairMSUCG_NEIGH::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g\n",i,epsilon[i][i],sigma[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairMSUCG_NEIGH::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g\n",i,j,epsilon[i][j],sigma[i][j],cut[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairMSUCG_NEIGH::single(int i, int j, int itype, int jtype, double rsq,
                         double factor_coul, double factor_lj,
                         double &fforce)
{
  error->all(FLERR, "Work not done here!"); /*---YP--- Haven't done this part yet */

  double r2inv,r6inv,forcelj,philj;

  r2inv = 1.0/rsq;
  r6inv = r2inv*r2inv*r2inv;
  forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
  fforce = factor_lj*forcelj*r2inv;

  philj = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) -
    offset[itype][jtype];
  return factor_lj*philj;
}

/* ---------------------------------------------------------------------- */

void *PairMSUCG_NEIGH::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str,"epsilon") == 0) return (void *) epsilon;
  if (strcmp(str,"sigma") == 0) return (void *) sigma;
  return NULL;
}
