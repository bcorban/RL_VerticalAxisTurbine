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
def f (N,n,l):
    print("process starts")
    g = gclib.py()
    c = g.GCommand
    g.GOpen("192.168.255.200 --direct -s ALL")
    t_1=time.time()
    for i in range(N):
        
        c("MG @AN[1],@AN[2],@AN[3],@AN[5],@AN[7]")
        print("reading")
        # n.value=i
        # l[:]=[i,1,2]
        if i%100==0 and i>0:
            print(i/(time.time()-t_1))
            
if __name__ == '__main__':
    # g = gclib.py()
    # c = g.GCommand
    # g.GOpen("192.168.255.200 --direct -s ALL")
    num=Value('i',0)
    manager=Manager()
    l=manager.list()
    p=Process(target=f, args=(10000,num,l))
    p.start()
    while num.value<10000:
        if num.value%100==0 and num.value>0:
            m=np.array(l)
            t_1=time.time()
            # g.GCommand(f"PAF={0}")
            # print(f"action takes {time.time()-t_1}")
            # print(f"pitch {(m,num.value)}")
            print(f"{m}",flush=True)

    p.join()

