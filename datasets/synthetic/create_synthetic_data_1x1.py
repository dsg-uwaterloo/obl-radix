# Based on https://zenodo.org/records/15169905
import random
import os

power_list = [26, 27, 28, 29, 30]
data_max = 100

for i in power_list:
    name = "./join_input_1x1_2power_"+str(i)+".txt"
    file = open(name, 'w')
    file.write(str(pow(2,i-1))+" "+str(pow(2,i-1))+"\n")
    file.write("\n")
    for o in range(pow(2,i-1)):
        file.write(str(o+1)+" "+str(random.randint(1, data_max))+"\n")
    file.write("\n")
    for o in range(pow(2,i-1)):
        file.write(str(o+1)+" "+str(random.randint(1, data_max))+"\n")
    file.close
    print(name + " ", end = "")
