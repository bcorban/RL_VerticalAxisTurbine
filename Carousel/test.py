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
def f (N,t_start):
    print("process starts")
    g = gclib.py()
    c = g.GCommand
    g.GOpen("192.168.255.200 --direct -s ALL")
    tpf=np.zeros(N)
    t=np.zeros(N)
    t_start.value=time.time()
    for i in range(N):
        
        tpf[i]=float(c("MG _TPF"))/(666.6666)
        t[i]=time.time()
    plt.figure()
    plt.scatter(t_a,6)
    plt.plot(t,tpf)
    
if __name__ == '__main__':
    g = gclib.py()
    c = g.GCommand
    g.GOpen("192.168.255.200 --direct -s ALL")
    t_start=Value('d',0)
    t_a=Value('d',0)
    p=Process(target=f, args=(1000,t_start))
    p.start()
    while p.is_alive:
        if t_start>0 and time.time-t_start.value>1:
            c(f"PAF {int(6*213.3333)}")
            t_a.value=time.time-t_start.value
            break
        
    p.join()

