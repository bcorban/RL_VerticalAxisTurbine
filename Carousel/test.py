import torch
import numpy as np
import time
import threading
import RL_ 
import gym
import matplotlib.pyplot as plt
import gclib
# from multiprocessing import Process, Value,Manager
import ctypes
from load_mat import load_mat
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from scipy import signal
fc_f = 20  # filter parameters
order_f = 4
fs = 550
fc_p = 30
order_p = 2
b, a = signal.butter(order_f, fc_f / (fs / 2)) 
print(a,b)
# MS = 213.3333
# ES = 666.6666


# def f (N,t_start,t_a):
#     print("process starts")
#     g = gclib.py()
#     c = g.GCommand
#     g.GOpen("192.168.255.200 --direct -s ALL")
#     tpf=np.zeros(N)
#     t=np.zeros(N)
#     t_start.value=time.time()
#     for i in range(N):
#         # print(i)
        
#         tpf[i]=float(c("MG _TPF"))/(ES)
#         t[i]=time.time()-t_start.value
#         time.sleep(0.0001)
#     np.savez('test.npz',
#              tpf=tpf,
#              t=t)
#     g.GClose()
    
# if __name__ == '__main__':
#     g = gclib.py()
#     c = g.GCommand
#     g.GOpen("192.168.255.200 --direct -s ALL")

#     ACF = 2
#     SPF = 1000000
#     KSF = 0.5  # Between 0.25 (low smoothing) and 64 (high smoothing)

#     c(f"ACF={round(256000 * ACF)}")
#     c(f"DCF={round(256000 * ACF)}")
#     c(f"SPF={SPF}")  # Default value: 10666
#     c(f"KSF={KSF}")  # Default value: 10666

#     g.GCommand("SHF")
#     g.GCommand("PTF=1")
#     g.GCommand("OEF=2")
#     t_start=Value('d',0)
#     t_a=Value('d',0)
#     p=Process(target=f, args=(500,t_start,t_a))
#     p.start()
#     while p.is_alive():
#         if t_start.value>0 and time.time()-t_start.value>1:
#             c(f"PAF={int(6*MS)}")
#             t_a.value=time.time()-t_start.value
#             break
    
#     p.join()
#     g.GCommand("PAF=0")
#     g.GCommand("OEF=0")
    
#     g.GClose()  
#     print(t_a.value)

if __name__ == '__main__':
    d=load_mat("/Users/PIVUSER/Desktop/RL_VerticalAxisTurbine/Carousel/2023_BC/bc001/raw/20230628/ms001mpt001_1.mat")
    history_volts=-(d["volts_raw_g"]-d["v_offset_g"])
    before=-(d["volts_raw_g"]-d["v_offset_g"])
    ws_f=10
    i_list=[]
    for i in range(11,5000):
        window = history_volts[i - ws_f : i + 1]
        med = np.median(window, axis=0)

        MAD = 1.4826 * np.median(
            np.abs(window - np.tile(med, (len(window), 1))), axis=0
        )  # inspired from matlab filloutliers method

        mask = np.abs(history_volts[i] - med) > 2 * MAD

        replace = np.where(
            mask, med + 2 * MAD * np.sign(history_volts[i] - med), history_volts[i]
        )
        replace = np.where(
            mask, med, history_volts[i]
        )


        history_volts[i] = replace

        flatlined = False

        for l in range(5): #check if any of the volts channels flatlined
            if np.all(history_volts[i - ws_f : i + 1, l] == med[l]):
                flatlined = True
                history_volts[i - ws_f : i + 1, l] = before[i - ws_f : i + 1, l]
                # print(f'flatlined {l}',flush=True)
                i_list.append(i)


    for k in range(5):            
        plt.figure()
        plt.plot(before[:i,k])
        plt.plot(history_volts[:i,k])

        plt.plot(i_list,np.zeros(len(i_list)),'o')
    plt.show()  