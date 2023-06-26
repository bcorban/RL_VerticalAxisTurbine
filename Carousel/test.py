import torch
import numpy as np
import time
import threading
import RL_ 
import gym
import matplotlib.pyplot as plt
import gclib
from multiprocessing import Process, Value,RawArray
import ctypes
def f (N,n):
    print("process starts")
    g = gclib.py()
    c = g.GCommand
    g.GOpen("192.168.255.200 --direct -s ALL")
    t_1=time.time()
    for i in range(N):
        
        c("MG @AN[1],@AN[2],@AN[3],@AN[5],@AN[7]")
        # print("reading")
        n.value=i
        if i%100==0 and i>0:
            print(i/(time.time()-t_1))
            
if __name__ == '__main__':
    g = gclib.py()
    c = g.GCommand
    g.GOpen("192.168.255.200 --direct -s ALL")
    num=Value('i',0)
    shared_array = RawArray(ctypes.c_double, 5)
    shared_array_np = np.ndarray(5,buffer=shared_array)
    p=Process(target=f, args=(10001,num))
    p.start()
    while num.value<10000:
        if num.value%100==0 and num.value>0:
            m=num.value
            t_1=time.time()
            g.GCommand(f"PAF={0}")
            # print(f"action takes {time.time()-t_1}")
            print(f"pitch {(m,num.value)}")

    p.join()

