import torch
import numpy as np
import time
import threading
import RL_ 
import gym
import matplotlib.pyplot as plt
import gclib
from multiprocessing import Process, Value

def f (N,n):
    print("process starts")
    g = gclib.py()
    c = g.GCommand
    g.GOpen("192.168.255.200 --direct -s ALL")
    for i in range(N):
        c("MG @AN[1]")
        print("reading")
        n.value=i
        
if __name__ == '__main__':
    g = gclib.py()
    c = g.GCommand
    g.GOpen("192.168.255.200 --direct -s ALL")
    num=Value('i',0)

    p=Process(target=f, args=(10001,num))
    p.start()
    while num.value<10000:
        if num.value%100==0 and num.value>0:
            m=num.value
            g.GCommand(f"PAF={0}")
            print(f"pitch {(m,num.value)}")

    p.join()

