import sys
import math
import numpy

# Read the filename from the command line arguments.
filename = sys.argv[1]

# initial value setup
wcut = 0.8
box = 42.9786  # isotropic box in x & y & z
g_end = 20.0
nhis = 201
delg = 0.1
x_high = 42.9786
x_low = 0.0
g_x = numpy.array([delg * (i + 0.5) for i in range(0, nhis)])
g_y = numpy.zeros(nhis)
g_vols = (4.0 / 3.0) * math.pi * numpy.array([(float(i + 1)**3.0 - float(i)**3.0) * (delg**3.0) for i in range(0, nhis)])
g_temp = numpy.zeros(nhis)
# Trajectory configuration variables
n_particles = 1000
neo_particles = 50
n_particles_read = 0
neo_particles_read = 0
pos_x = numpy.zeros(n_particles)
pos_y = numpy.zeros(n_particles)
pos_z = numpy.zeros(n_particles)
pos_neo_x = numpy.zeros(neo_particles)
pos_neo_y = numpy.zeros(neo_particles)
pos_neo_z = numpy.zeros(neo_particles)
w = numpy.zeros(n_particles)
# Unwrap function
def unwrap(pos, xlo, xhi):
    returnvalue = pos* (xhi-xlo) + xlo
    return returnvalue

# Other control flow variables
zero_flag = 0
update = 0
average_number = 0.0
poi_before = 0.0
time_element = 0
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
        if (time_element < 6000000):
            if(line_elements[0] == 'ITEM:'):
                header_line_counter = (header_line_counter + 1) % 4
            # If it is not a header line, read in the configuration.
            if(time_element > -1):
                if(header_line_counter == 0 and line_elements[0] != 'ITEM:' and int(line_elements[0]) <= 1000):
                    pos_x[n_particles_read] = unwrap(float(line_elements[2]), 0.0, box)
                    pos_y[n_particles_read] = unwrap(float(line_elements[3]), 0.0, box)
                    pos_z[n_particles_read] = unwrap(float(line_elements[4]), 0.0, box)
                    w[n_particles_read] = float(line_elements[11])
                    n_particles_read += 1
                elif(header_line_counter == 0 and line_elements[0] != 'ITEM:' and int(line_elements[0]) > 1000):
                    pos_neo_x[neo_particles_read] = unwrap(float(line_elements[2]), 0.0, box)
                    pos_neo_y[neo_particles_read] = unwrap(float(line_elements[3]), 0.0, box)
                    pos_neo_z[neo_particles_read] = unwrap(float(line_elements[4]), 0.0, box)
                    neo_particles_read += 1
            # If it is a timestep header line, read it and process
            # the previous frame.
            if(header_line_counter == 1 and len(line_elements) == 1):
                time_element = int(line_elements[0])
                print(time_element)
                if (n_particles_read > 0):
                    g_temp = numpy.zeros(nhis)
                    poi = 0.0
                    poj = 0.0
                    for i in range(0, neo_particles-1):
                        poi += 1.0
                        for j in range(i+1, neo_particles):
                            if(i != j):
                                while (pos_neo_x[i] < 0.0 or pos_neo_x[i] > box):
                                    if (pos_neo_x[i] < 0.0):
                                        pos_neo_x[i] += box
                                    else:
                                        pos_neo_x[i] -= box
                                while (pos_neo_y[i] < 0.0 or pos_neo_y[i] > box):
                                    if (pos_neo_y[i] < 0.0):
                                        pos_neo_y[i] += box
                                    else:
                                        pos_neo_y[i] -= box
                                while (pos_neo_z[i] < 0.0 or pos_neo_z[i] > box):
                                    if (pos_neo_z[i] < 0.0):
                                        pos_neo_z[i] += box
                                    else:
                                        pos_neo_z[i] -= box
                                while (pos_neo_x[j] < 0.0 or pos_neo_x[j] > box):
                                    if (pos_neo_x[j] < 0.0):
                                        pos_neo_x[j] += box
                                    else:
                                        pos_neo_x[j] -= box
                                while (pos_neo_y[j] < 0.0 or pos_neo_y[j] > box):
                                    if (pos_neo_y[j] < 0.0):
                                        pos_neo_y[j] += box
                                    else:
                                        pos_neo_y[j] -= box
                                while (pos_neo_z[j] < 0.0 or pos_neo_z[j] > box):
                                    if (pos_neo_z[j] < 0.0):
                                        pos_neo_z[j] += box
                                    else:
                                        pos_neo_z[j] -= box
                                pos_neo_dx = pos_neo_x[i] - pos_neo_x[j]
                                pos_neo_dx = pos_neo_dx - box * round(pos_neo_dx / box)
                                pos_neo_dy = pos_neo_y[i] - pos_neo_y[j]
                                pos_neo_dy = pos_neo_dy - box * round(pos_neo_dy / box)
                                pos_neo_dz = pos_neo_z[i] - pos_neo_z[j]
                                pos_neo_dz = pos_neo_dz - box * round(pos_neo_dz / box)
                                dist = (pos_neo_dx * pos_neo_dx + pos_neo_dy * pos_neo_dy + pos_neo_dz * pos_neo_dz)**(0.5)
                                if dist < (g_end):
                                    ig = int(dist / delg)
                                    g_temp[ig] += 2.0
                    print poi
                    if (poi > 1.0):
                        number_density = (poi - 1.0) / (box * box * box)
                        print(number_density)
                        average_number  = poi
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
                    neo_particles_read = 0
                    pos_x = numpy.zeros(n_particles)
                    pos_y = numpy.zeros(n_particles)
                    pos_z = numpy.zeros(n_particles)
                    pos_neo_x = numpy.zeros(neo_particles)
                    pos_neo_y = numpy.zeros(neo_particles)
                    pos_neo_z = numpy.zeros(neo_particles)
                    w = numpy.zeros(n_particles)
                    update = 1
                if (average_number > 0 and update == 1):
                    update = 0
                    if (zero_flag == 0):
                        g_y = ( poi_before * g_y + average_number * g_temp) / float(average_number+poi_before)
                        poi_before += average_number
                        print(average_number)
        else:
            break

out_filename = "rdf_NN.data"
with open(out_filename, 'w') as rdf_data:
    for i in range(0, nhis):
        tmp = str(g_x[i]) + ' ' + str(g_y[i])
        rdf_data.write(tmp)
        rdf_data.write('\n')
