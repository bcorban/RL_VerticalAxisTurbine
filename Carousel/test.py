import torch
import numpy as np
import time
import threading
import RL_ 
import gym
import matplotlib.pyplot as plt
import gclib
# print(torch.cuda.is_available())

# gpu = torch.device('cuda:0') 

# cpu = torch.device('cpu')

# N = 5000

# # operations sur cpu
# xc = torch.tensor(np.random.normal(size = (N,N)), device = cpu)
# yc = torch.tensor(np.random.normal(size = (N,N)), device = cpu)

# t0 = time.time()
# zc = yc
# for k in range(1000):
#     zc = xc*zc
# zc = torch.sum(zc)
# print(zc)
# print(time.time() - t0)
# print(zc.device)

# # operations sur gpu
# xg = torch.tensor(np.random.normal(size = (N,N)), device = gpu)
# yg = torch.tensor(np.random.normal(size = (N,N)), device = gpu)

# t1 = time.time()
# zg = yg
# for k in range(1000):
#     zg = xg*zg
# zg = torch.sum(zg)
# print(zg)
# print(time.time() - t1)
# print(zg.device)

# print()


g = gclib.py()
c = g.GCommand
g.GOpen("192.168.255.200 --direct -s ALL")

env=gym.make("RL_/CustomEnv-v0")

env.reset()
def continuously_read():
    t=time.time()
    while time.time()-t<10:
        env.read_state()
daemon=threading.Thread(target=continuously_read,daemon=True,name="state_reader")
save=[]
ti=time.time()
daemon.start()
while time.time()-ti<12:
    save.append(env.i)
    time.sleep(0.5)
print(save)

print()