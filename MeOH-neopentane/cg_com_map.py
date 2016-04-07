import copy
import math
import sys
import numpy as np

# Initial variables
filename = sys.argv[1]
header = filename.split(".")
outputname = "cg.lammpstrj"
f = open(filename, 'r')
out_data = open(outputname, 'w')
k = 0  # Needed to parsing the lammpstrj
count_time = 0
box_size = 43.11145  # Volume is preserved during a NVT simulation and retrieved from lammpstrj file
index = [i for i in range(0, 6850)]  # 6850 : number of total atoms in the system
index_temp = []
coord = []
force = []
time_element = 0

# PBC consideration
xlo = xhi = ylo = yhi = zlo = zhi = 0.0
axis_read = 0  # Used to read the axis information from trj file for each frame

# Wrapping function
def scale_position(pos, lo, hi):
    scaled_pos = pos * (hi-lo) + lo
    return(scaled_pos)

def distance(a, b):
    dist = ((a[0]-b[0])**2.0 + (a[1]-b[1])**2.0 + (a[2]-b[2])**2.0)**0.5
    return(dist)
###########################################################
############### Trajectory Processing ...  ################
###########################################################

for line in f:
    line_element = line.split()
    if(line_element[0] == 'ITEM:'):
        k = (k+1) % 4
        if (len(line_element) != 13):
            out_data.write(line)
        else:
            no_velocity = 'ITEM: ATOMS id type x y z fx fy fz\n'
            out_data.write(no_velocity)
    if(k == 1 and len(line_element) == 1):
        time_element = int(line_element[0])
        index_temp = []
        coord = []
        force = []
        out_data.write(line)
    if(k == 2 and len(line_element) == 1):
        n_mole = "1050\n"
        out_data.write(n_mole)
    if(k == 3 and len(line_element) == 2):
        if (axis_read == 0):
            xlo = float(line_element[0])
            xhi = float(line_element[1])
            axis_read += 1
        if (axis_read == 1):
            ylo = float(line_element[0])
            yhi = float(line_element[1])
            axis_read += 1
        if (axis_read == 2):
            zlo = float(line_element[0])
            zhi = float(line_element[1])
            axis_read = 0
        pbc_str = "0.00000 43.11145\n"
        out_data.write(pbc_str)
    if(k == 0 and line_element[0] != 'ITEM:'):  # Scan for all
        # In this file, we know that the input lammpstrj file is sorted by id
        # (dump 1 sort id) so we can directly append it to coordinates and
        # process it to calculate COM b/c packmol gives you the right order
        coord.append([float(line_element[2]), float(line_element[3]), float(line_element[4])])
        force.append([float(line_element[8]), float(line_element[9]), float(line_element[10])])
        index_temp.append(int(line_element[0]))  # Count the # of atoms (each frame)
        if (len(index_temp) == 6850):  # Start the mapping process
            # Initialization
            com_methanol = []
            com_neopentane = []
            com_methanol_force = []
            com_neopentane_force = []
            size_case = len(coord)  # Sorting by the atom id is not needed.
            print_arg = "Time step: %d is starting! with %d %f %f %f \n" % (time_element, size_case, xlo, ylo, zlo)
            print(print_arg)
            # Molecule-wise COM
            # First case: methanol (1~6000 atoms i.e. 1~1000 molecules)
            for i in range(0, 1000):
                methanol = []
                methanol_force = []
                # Append to the methanol array
                for j in range(0, 6):
                    temp = [scale_position(coord[6*i+j][0], xlo, xhi), scale_position(coord[6*i+j][1], ylo, yhi), scale_position(coord[6*i+j][2], zlo, zhi)]
                    methanol.append(temp)
                    temp_force = [force[6*i+j][0], force[6*i+j][1], force[6*i+j][2]]
                    methanol_force.append(temp_force)
                # Calculate the delta postition (fix the first particle as the origin)
                dx = []
                for j in range(1, 6):
                    delta = [methanol[j][0]-methanol[0][0], methanol[j][1]-methanol[0][1], methanol[j][2]-methanol[0][2]]
                    for ii in range(0, 3):
                        while(delta[ii] > box_size/2.0 or delta[ii] < -0.5 * box_size):  # PBC consideration
                            if delta[ii] > box_size/2.0:
                                delta[ii] -= box_size
                            elif delta[ii] < -0.5 * box_size:
                                delta[ii] += box_size
                    dx.append(delta)
                # COM calculation
                methanol_unwrap = []
                methanol_unwrap.append([methanol[0][0], methanol[0][1], methanol[0][2]])
                for j in range(1, 6):
                    unwrap_temp = [methanol[0][0] + dx[j-1][0], methanol[0][1] + dx[j-1][1], methanol[0][2] + dx[j-1][2]]
                    methanol_unwrap.append(unwrap_temp)
                com = (12.011000*np.array(methanol_unwrap[0]) + 1.008000*(np.array(methanol_unwrap[1])+np.array(methanol_unwrap[2])+np.array(methanol_unwrap[3])+np.array(methanol_unwrap[5]))+15.999000*np.array(methanol_unwrap[4]))/32.04200
                com_meoh = (np.array(methanol_force[0]) + np.array(methanol_force[1])+np.array(methanol_force[2])+np.array(methanol_force[3])+np.array(methanol_force[5])+np.array(methanol_force[4]))
                for jj in range(0, 3):
                    while (com[jj] > box_size or com[jj] < 0.0):
                        if com[jj] > box_size:
                            com[jj] -= box_size
                        elif com[jj] < 0.0:
                            com[jj] += box_size
                # PBC wrap
                com_methanol.append(com)
                com_methanol_force.append(com_meoh)
            # Molecule-wise COM
            # Second case: neopentane (6001~6850 atoms i.e. 1~50 molecules)
            for i in range(0, 50):
                neopentane = []
                neopentane_force = []
                # Append to the neopentane array
                for j in range(0, 17):
                    temp = [scale_position(coord[17*i+j+6000][0], xlo, xhi), scale_position(coord[17*i+j+6000][1], ylo, yhi), scale_position(coord[17*i+j+6000][2], zlo, zhi)]
                    neopentane.append(temp)
                    temp_force = [force[17*i+j+6000][0], force[17*i+j+6000][1], force[17*i+j+6000][2]]
                    neopentane_force.append(temp_force)
                # Calculate the delta postition (fix the first particle as the origin)
                dx = []
                for j in range(0, 17):
                    delta = [neopentane[j][0]-neopentane[0][0], neopentane[j][1]-neopentane[0][1], neopentane[j][2]-neopentane[0][2]]
                    for ii in range(0, 3):
                        while(delta[ii] > box_size/2.0 or delta[ii] < -0.5 * box_size):  # PBC consideration
                            if delta[ii] > box_size/2.0:
                                delta[ii] -= box_size
                            elif delta[ii] < -0.5 * box_size:
                                delta[ii] += box_size
                    dx.append(delta)
                # COM calculation
                neopentane_unwrap = []
                neopentane_unwrap.append(neopentane[0])
                for j in range(1, 17):
                    unwrap_temp = [neopentane[0][0] + dx[j-1][0],neopentane[0][1] + dx[j-1][1],neopentane[0][2] + dx[j-1][2]]
                    neopentane_unwrap.append(unwrap_temp)
                com_final = (12.011000 * (np.array(neopentane_unwrap[0])+np.array(neopentane_unwrap[4])+np.array(neopentane_unwrap[5])+np.array(neopentane_unwrap[9])+np.array(neopentane_unwrap[13])) + 1.008000*(np.array(neopentane_unwrap[1])+np.array(neopentane_unwrap[2])+np.array(neopentane_unwrap[3])+np.array(neopentane_unwrap[6])+np.array(neopentane_unwrap[7])+np.array(neopentane_unwrap[8])+np.array(neopentane_unwrap[10])+np.array(neopentane_unwrap[11])+np.array(neopentane_unwrap[12])+np.array(neopentane_unwrap[14])+np.array(neopentane_unwrap[15])+np.array(neopentane_unwrap[16])))/72.15100
                com_neo = (np.array(neopentane_force[0])+np.array(neopentane_force[4])+np.array(neopentane_force[5])+np.array(neopentane_force[9])+np.array(neopentane_force[13]) + np.array(neopentane_force[1])+np.array(neopentane_force[2])+np.array(neopentane_force[3])+np.array(neopentane_force[6])+np.array(neopentane_force[7])+np.array(neopentane_force[8])+np.array(neopentane_force[10])+np.array(neopentane_force[11])+np.array(neopentane_force[12])+np.array(neopentane_force[14])+np.array(neopentane_force[15])+np.array(neopentane_force[16]))
                for jj in range(0, 3):
                    while (com_final[jj] > box_size or com_final[jj] < 0.0):
                        if com_final[jj] > box_size:
                            com_final[jj] -= box_size
                        elif com_final[jj] < 0.0:
                            com_final[jj] += box_size
                com_neopentane.append(com_final)
                com_neopentane_force.append(com_neo)
            # Print out the CG-ed sites
            for i in range(0, 1000):
                methanol_output = str(i+1) + ' 1 %.6f %.6f %.6f %2.10f %2.10f %2.10f\n' % (com_methanol[i][0], com_methanol[i][1], com_methanol[i][2], com_methanol_force[i][0], com_methanol_force[i][1], com_methanol_force[i][2])
                out_data.write(methanol_output)
            for i in range(0, 50):
                neopentane_output = str(i+1001) + ' 2 %.6f %.6f %.6f %2.10f %2.10f %2.10f\n' % (com_neopentane[i][0], com_neopentane[i][1], com_neopentane[i][2], com_neopentane_force[i][0], com_neopentane_force[i][1], com_neopentane_force[i][2])
                out_data.write(neopentane_output)
            # Clean up the variables
            coord = []
            index_temp = []
            print_arg = "Time step: %d is done! \n" % time_element
            print(print_arg)
f.close()
out_data.close()
