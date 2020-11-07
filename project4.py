# Project 4 - HSPICE
# NAME: SIVANAGA SURYA VAMSI POPURI
# ASU ID: 1217319207

import numpy as np
import subprocess
import shutil # for copying files


def comm(N, fan, file):
    shutil.copy(file, this_file)
    h_file = open(this_file, "a")
    n_letter = ord('a') # letter for the first node  
    line =  ".measure TRAN tphl_inv TRIG v(Xinv1.a) VAL = 1.5 RISE = 1 TARG v(Xinv)"
    line += str(N) + ".z) VAL=1.5 FALL = 1/n"
    h_file.write(line)
    
    h_file.write(".param fan = " + str(fan) + "\n")
    
    for i in range(1, N+1):
        line = "Xinv" + str(n) + " " + chr(n_letter) + " "
        if i == N:
            line += "z inv M="
        else:
            n_letter += 1
            line += chr(n_letter) + " inv M="
        
        if i == 1:
            line += "1/n"
        else:
            line += ((i-2)*"fan*") + "fan\n"
        h_file.write(line)
    
    h_file.write(".end")
    h_file.close()
    return h_file

file = open("InvChain.sp", 'r')
N = np.arange(1, 15, 2) # number of inverters needs to be odd
fan = np.arange(2, 15, 1) 
delay_list = []
param_list = []

for n in range(N):
    for f in range(fan):
        p_list = [n, f]
        param_list.append(p_list)
        h_file = comm(n, f, file)
        proc = subprocess.Popen(["hspice",h_file],
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        output, err = proc.communicate()
        # extract tphl from the output file
        data = np.recfromcsv("InvChain.mt0.csv",comments="$",skip_header=3)
        tphl = data["tphl_inv"] # the delay
        delay_list.append(tphl)
        print("Delay for %d inverters and %d fan value is %f"%(n, f, tphl))