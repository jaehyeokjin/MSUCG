import sys
import math
import numpy

# Read the filename from the command line arguments.
filename = sys.argv[1]

# initial value setup
wcut = 10.0
box = 10.700  # isotropic box in x & y
z_box = 21.400  # twice the length in z
g_end = 10.0
nhis = 101
delg = 0.1
x_high = 10.7
x_low = 0.0
g_x = numpy.array([delg * (i + 0.5) for i in range(0, nhis)])
g_y = numpy.zeros(nhis)
g_vols = (4.0 / 3.0) * math.pi * numpy.array([(float(i + 1)**3.0 - float(i)**3.0) * (delg**3.0) for i in range(0, nhis)])
g_temp = numpy.zeros(nhis)
# Trajectory configuration variables
n_particles = 1125
n_particles_read = 0
pos_x = numpy.zeros(n_particles)
pos_y = numpy.zeros(n_particles)
pos_z = numpy.zeros(n_particles)
w = numpy.zeros(n_particles)
# Other control flow variables
zero_flag = 0
update = 0
average_number = 0

# Open & read in the file for analysis.
with open(filename, "r") as filestream:
    # Define two trajectory reading helper variables.
    header_line_counter = 0
    time_element = 0
    # Read the file line by line, gathering
    # input & analyzing it frame by frame.
    for line in filestream:
        line_elements = line.split()
        # Determine if this line is in a header section & where,
        # if so.
        if(line_elements[0] == 'ITEM:'):
            header_line_counter = (header_line_counter + 1) % 4
        # If it is not a header line, read in the configuration.
        if(time_element > 80000):
            if(time_element % 1000 == 0 and header_line_counter == 0 and line_elements[0] != 'ITEM:'):
                pos_x[n_particles_read] = float(line_elements[2])
                pos_y[n_particles_read] = float(line_elements[3])
                pos_z[n_particles_read] = float(line_elements[4])
                w[n_particles_read] = float(line_elements[10])
                n_particles_read += 1
        # If it is a timestep header line, read it and process
        # the previous frame.
        if(header_line_counter == 1 and len(line_elements) == 1):
            time_element = int(line_elements[0])
            print(time_element)
            if (n_particles_read > 0):
                g_temp = numpy.zeros(nhis)
                poi = 0.0
                poj = 0.0
                for i in range(0, n_particles):
                    tan_factor = math.tanh((w[i] - wcut) / (0.1 * wcut))
                    poi += 1.0
                    poj += 1.0
                    for j in range(0, n_particles):
                        if(i != j):
                            pos_dx = pos_x[i] - pos_x[j]
                            pos_dx = pos_dx - box * round(pos_dx / box)
                            pos_dy = pos_y[i] - pos_y[j]
                            pos_dy = pos_dy - box * round(pos_dy / box)
                            pos_dz = pos_z[i] - pos_z[j]
                            pos_dz = pos_dz - box * round(pos_dz / z_box)
                            dist = (pos_dx * pos_dx + pos_dy *
                                    pos_dy + pos_dz * pos_dz)**(0.5)
                            if dist < (box / 2.0):
                                ig = int(dist / delg)
                                g_temp[ig] += 1.0

                print poi
                if (poi > 1.0):
                    number_density = (poi - 1.0) / (box * box * box * 2.0)
                    print(number_density)
                    average_number += 1
                # Normalize the temp histogram if it is nonzero.
                # Flag if it is zero. (In this case that is impossible.)
                if (poi > 1.0):
                    num_ideal = number_density * g_vols
                    g_temp = (g_temp / num_ideal) / float(poi)
                    zero_flag = 0
                else:
                    zero_flag = 1
                # Reset the frame data now that it has been
                # analyzed.
                n_particles_read = 0
                pos_x = numpy.zeros(n_particles)
                pos_y = numpy.zeros(n_particles)
                pos_z = numpy.zeros(n_particles)
                w = numpy.zeros(n_particles)
                update = 1
            if (average_number > 0 and update == 1):
                update = 0
                if (zero_flag == 0):
                    g_y = (float(average_number - 1) * g_y + g_temp) / float(average_number)
    print(average_number)

filename_stem = filename.split(".")[0]
out_filename = filename_stem + "_rdf.data"
with open(out_filename, 'w') as rdf_data:
    for i in range(0, nhis):
        tmp = str(g_x[i]) + ' ' + str(g_y[i])
        rdf_data.write(tmp)
        rdf_data.write('\n')
