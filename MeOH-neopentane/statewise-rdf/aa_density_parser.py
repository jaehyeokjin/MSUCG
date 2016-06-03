import copy
import math
import sys
import numpy as np
import random

# Initial variables
outputname = "cg_w_density.lammpstrj"
inputname = "cg.lammpstrj"
f = open(inputname, 'r')
out_data = open(outputname, 'w')
k = 0
count_time = 0
index = [i for i in range(0, 1050)]
index_temp = []
coord_neo = []
coord_meoh= []
force = []
time_element = 0
# UCG treatment
r_th = 7.0
w_th = 0.8
box_size = 42.97861

# Pre-defined string (in order to save the time for processing)
str_a = "ITEM: TIMESTEP\n"
str_b = "\nITEM: NUMBER OF ATOMS\n1050\nITEM: BOX BOUNDS pp pp pp\n0.00000 43.11145\n0.00000 43.11145\n0.00000 43.11145\nITEM: ATOMS id type x y z vx vy vz fx fy fz c_5 c_4\n" # c_4 is dummy variable in here - but remained as consistent with rdf processing code
###########################################################
############### Trajectory Processing ...  ################
###########################################################

for line in f:
    line_element = line.split()
    if(line_element[0] == 'ITEM:'):
        k = (k+1) % 4
    if(k == 1 and len(line_element) == 1):
        time_element = int(line_element[0])
        index_temp = []
        coord_meoh= []
        coord_neo = []
        force = []
        line_neopentane = ""
    ''' This part is not needed in the post-processed cg trj
    if(k == 2 and len(line_element) == 1):
        out_data.write(line)
    if(k == 3 and len(line_element) == 2):
        out_data.write(line)
    '''
    if(k == 0 and line_element[0] != 'ITEM:'):  # Scan for all
        index_temp.append(int(line_element[0])) # Until the one time frame is successfully parsed
        if ( len(index_temp) > 1000 ):
            coord_neo.append([float(line_element[2]), float(line_element[3]), float(line_element[4])])
            # Neopentane doesn't have any states so just dump it in the line_neopentane string variable.
            line_neopentane += line_element[0] + " 3"
            for iter_line in range(2,5):
                line_neopentane += " " + line_element[iter_line]
            line_neopentane += " 0.0 0.0 0.0"
            for iter_line in range(5,8):
                line_neopentane += " " + line_element[iter_line]
            line_neopentane += " 0.0 0.0\n" # Neopentane doesn't have c_5 c_4 values
        else:
            coord_meoh.append([float(line_element[2]), float(line_element[3]), float(line_element[4])])
        force.append([float(line_element[5]), float(line_element[6]), float(line_element[7])]) # Force is not changed
        if (len(index_temp) == 1050):  # After parsing, construct the molecule-wise distance
            dist_m = [[0.0 for _ in range(0,50)] for _ in range(0,1000)] # Distance matrix initialization: dimension (MeOH X Neopentane) 
            for i in range(0, 1000):
                for j in range(0, 50):
                    delta = [coord_meoh[i][0] - coord_neo[j][0], coord_meoh[i][1] - coord_neo[j][1], coord_meoh[i][2] - coord_neo[j][2]]
                    for ii in range(0, 3):
                        while(delta[ii] > box_size/2.0 or delta[ii] < -0.5 * box_size):
                            if delta[ii] > box_size/2.0:
                                delta[ii] -= box_size
                            elif delta[ii] < -0.5 * box_size:
                                delta[ii] += box_size
                    dist_m[i][j] = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]) ** 0.5 # Update the distance
            # Density calculation and update it as the probability
            local_density = [0.0 for _ in range(0,1000)]
            local_prob = [0.0 for _ in range(0,1000)]
            for i in range(0, 1000):
                local_density_on_i = 0.0
                for j in range(0, 50):
                    local_density_on_i += 0.5 * (1.0 - math.tanh((dist_m[i][j] - r_th)/(0.1 * r_th)))
                local_density[i] = local_density_on_i
            # Construct the repetitive replica of the trajectory (as much as n_traj in the above code)
            print_statement = str_a + str(time_element) + str_b
            out_data.write(print_statement)
            for i in range(0, 1000):
                # Print the methanol-only output (neopentane is written in line_neopentane)
                line_methanol = str(i+1) + ' 1' + ' %.6f %.6f %.6f 0.0 0.0 0.0 %2.10f %2.10f %2.10f %2.10f 0.0\n' % (coord_meoh[i][0], coord_meoh[i][1], coord_meoh[i][2], force[i][0], force[i][1], force[i][2], local_density[i])
                out_data.write(line_methanol)
            out_data.write(line_neopentane)

            # Clean up the variables
            index_temp= []
            coord_meoh= []
            coord_neo = []
            print_arg = "Time step: %d is done! \n" % time_element
            print(print_arg)

f.close()
out_data.close()