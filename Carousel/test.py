import torch
import numpy as np
import time
import threading
import RL_ 
import gym
import matplotlib.pyplot as plt
import gclib
from multiprocessing import Process, Value,Manager
import ctypes


MS = 213.3333
ES = 666.6666


def f (N,t_start,t_a):
    print("process starts")
    g = gclib.py()
    c = g.GCommand
    g.GOpen("192.168.255.200 --direct -s ALL")
    tpf=np.zeros(N)
    t=np.zeros(N)
    t_start.value=time.time()
    for i in range(N):
        # print(i)
        
        tpf[i]=float(c("MG _TPF"))/(ES)
        t[i]=time.time()-t_start.value
        time.sleep(0.0001)
    np.savez('test.npz',
             tpf=tpf,
             t=t)
    g.GClose()
    
if __name__ == '__main__':
    g = gclib.py()
    c = g.GCommand
    g.GOpen("192.168.255.200 --direct -s ALL")

    ACF = 2
    SPF = 1000000
    KSF = 0.5  # Between 0.25 (low smoothing) and 64 (high smoothing)

    c(f"ACF={round(256000 * ACF)}")
    c(f"DCF={round(256000 * ACF)}")
    c(f"SPF={SPF}")  # Default value: 10666
    c(f"KSF={KSF}")  # Default value: 10666

    g.GCommand("SHF")
    g.GCommand("PTF=1")
    g.GCommand("OEF=2")
    t_start=Value('d',0)
    t_a=Value('d',0)
    p=Process(target=f, args=(500,t_start,t_a))
    p.start()
    while p.is_alive():
        if t_start.value>0 and time.time()-t_start.value>1:
            c(f"PAF={int(6*MS)}")
            t_a.value=time.time()-t_start.value
            break
    
    p.join()
    g.GCommand("PAF=0")
    g.GCommand("OEF=0")
    
    g.GClose()  
    print(t_a.value)



