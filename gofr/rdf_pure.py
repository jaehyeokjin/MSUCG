import sys
import copy
import math
filename = sys.argv[1]
name=filename.split(".")
f = open(filename,"r")
outname = name[0]+"_rdf.data"
rdf_data=open(outname,'w')
lines = f.readlines()
k = 0
time_element=0
# float_array: string array -> float array
def float_array(input):
        result = []
        for i in range(len(input)):
                result.append(float(input[i]))
        return result

# initial value setup
zero_flag = 0
wcut = 10.0
box = 10.700 # isotropic box
z_box = 21.400
g_end = 10.0
ngr = 0
nhis = 101
delg = 0.1
x_high = 10.7
x_low = 0.0
g_x=[i for i in range(1,nhis+1)]
g_y=[0 for _ in range(1,nhis+1)]
g_temp=[0 for _ in range(1,nhis+1)]
pos_x = []
pos_y = []
pos_z = []
w = []
update = 0
average_number = 0
for i in range(0,nhis):
        g_y[i] = 0.0

# Counting the number of the particles for each case and put it in count[] array
# 1 : refers to normal oxygen
# 2 : refers to normal hydrogen
# 3 : refers to oxygen from hydronium
# 4 : refers to hydrogen from hydronium
# 5 : Cl- ion
index = 1 # test case for normal oxygens
count = 0 # number of the particle
for line in lines:
        line_element=line.split()
        if(line_element[0] == 'ITEM:'):
            k = (k+1) % 4
        if(k == 1 and len(line_element) == 1):
            time_element=int(line_element[0])
            print(time_element)
            if (len(pos_z) > 0):
                for i in range(0,nhis):
                    g_temp[i] = 0.0 
                pos_x = float_array(pos_x)
                pos_y = float_array(pos_y)
                pos_z = float_array(pos_z)
                w = float_array(w)
                poi = 0.0
                poj = 0.0

                for i in range(0,len(pos_x)):
                    tan_factor = math.tanh((w[i]-wcut)/(0.1*wcut))
                    poi += 1.0
                    poj += 1.0
                for i in range(0,len(pos_x)):
                    for j in range(0,len(pos_x)):
                        if(i != j):
                            pos_dx=pos_x[i]-pos_x[j]
                            pos_dx=pos_dx-box*round(pos_dx/box)
                            pos_dy=pos_y[i]-pos_y[j]
                            pos_dy=pos_dy-box*round(pos_dy/box)
                            pos_dz=pos_z[i]-pos_z[j]
                            pos_dz=pos_dz-box*round(pos_dz/z_box)
                            dist=(pos_dx*pos_dx+pos_dy*pos_dy+pos_dz*pos_dz)**(0.5)
                            if dist < (box/2.0):
                                ig=int(dist/delg)
                                g_temp[ig]=g_temp[ig]+1.0
                print(poi)
                if (poi > 1.0):
                    rho = (poi-1.0)/(box * box * box * 2.0)
                    print(rho)
                    average_number += 1
                for i in range(0,nhis):
                    if (poi > 1.0):
                        r = delg*(i+0.5)
                        #volume = ((i+1)**2.0 - i**2.0 ) * (delg**2.0)
                        volume = (float(i+1)**3.0 - float(i)**3.0) *(delg**3.0)
                        num_ideal = (4.0/3.0)*math.pi*volume*rho
                        #num_ideal = math.pi*volume*rho
                        g_temp[i] = g_temp[i] / (float(poi)*num_ideal)
                        g_x[i] = r
                        zero_flag = 0
                    else:
                        g_temp[i] = 0.0
                        zero_flag = 1
                pos_x = []
                pos_y = []    
                pos_z = []
                w = []
                update = 1
            if (average_number > 0 and update == 1):
                update = 0
                for i in range(0,len(g_temp)):
                    if (zero_flag == 0):
                        g_y[i] = (g_y[i] * float(average_number-1) + g_temp[i])/float(average_number)
        if(time_element > 80000):
            if(time_element % 1000 == 0 and k == 0 and line_element[0] != 'ITEM:'):
                pos_x.append(line_element[2])
                pos_y.append(line_element[3])
                pos_z.append(line_element[4])
                w.append(line_element[10])
print(average_number)
for i in range(1,nhis+1):
	tmp=str(g_x[i-1])+' '+str(g_y[i-1])
	rdf_data.write(tmp)
	rdf_data.write('\n')
f.close()
rdf_data.close()
