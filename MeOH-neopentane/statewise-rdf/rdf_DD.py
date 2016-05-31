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
n_particles_read = 0
pos_x = numpy.zeros(n_particles)
pos_y = numpy.zeros(n_particles)
pos_z = numpy.zeros(n_particles)
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
        if (time_element < 1000000):
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
            # If it is a timestep header line, read it and process
            # the previous frame.
            if(header_line_counter == 1 and len(line_elements) == 1):
                time_element = int(line_elements[0])
                print(time_element)
                if (n_particles_read > 0):
                    g_temp = numpy.zeros(nhis)
                    poi = 0.0
                    poj = 0.0
                    for i in range(0, n_particles-1):
                        tan_factor = math.tanh((w[i] - wcut) / (0.1 * wcut))
                        poi += 0.5*(1.0+tan_factor)
                        for j in range(i+1, n_particles):
                            if(i != j):
                                while (pos_x[i] < 0.0 or pos_x[i] > box):
                                    if (pos_x[i] < 0.0):
                                        pos_x[i] += box
                                    else:
                                        pos_x[i] -= box
                                while (pos_y[i] < 0.0 or pos_y[i] > box):
                                    if (pos_y[i] < 0.0):
                                        pos_y[i] += box
                                    else:
                                        pos_y[i] -= box
                                while (pos_z[i] < 0.0 or pos_z[i] > box):
                                    if (pos_z[i] < 0.0):
                                        pos_z[i] += box
                                    else:
                                        pos_z[i] -= box
                                while (pos_x[j] < 0.0 or pos_x[j] > box):
                                    if (pos_x[j] < 0.0):
                                        pos_x[j] += box
                                    else:
                                        pos_x[j] -= box
                                while (pos_y[j] < 0.0 or pos_y[j] > box):
                                    if (pos_y[j] < 0.0):
                                        pos_y[j] += box
                                    else:
                                        pos_y[j] -= box
                                while (pos_z[j] < 0.0 or pos_z[j] > box):
                                    if (pos_z[j] < 0.0):
                                        pos_z[j] += box
                                    else:
                                        pos_z[j] -= box
                                pos_dx = pos_x[i] - pos_x[j]
                                pos_dx = pos_dx - box * round(pos_dx / box)
                                pos_dy = pos_y[i] - pos_y[j]
                                pos_dy = pos_dy - box * round(pos_dy / box)
                                pos_dz = pos_z[i] - pos_z[j]
                                pos_dz = pos_dz - box * round(pos_dz / box)
                                dist = (pos_dx * pos_dx + pos_dy * pos_dy + pos_dz * pos_dz)**(0.5)
                                tan_i = math.tanh((w[i] - wcut) / (0.1 * wcut))
                                tan_j = math.tanh((w[j] - wcut) / (0.1 * wcut))
                                p_i = 0.5*(1.0+tan_i)
                                p_j = 0.5*(1.0+tan_j)
                                if dist < (g_end):
                                    ig = int(dist / delg)
                                    g_temp[ig] += p_i * p_j * 2.0
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
                    pos_x = numpy.zeros(n_particles)
                    pos_y = numpy.zeros(n_particles)
                    pos_z = numpy.zeros(n_particles)
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

out_filename = "rdf_DD.data"
with open(out_filename, 'w') as rdf_data:
    for i in range(0, nhis):
        tmp = str(g_x[i]) + ' ' + str(g_y[i])
        rdf_data.write(tmp)
        rdf_data.write('\n')
