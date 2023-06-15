import torch
import numpy as np
import time
print(torch.cuda.is_available())

gpu = torch.device('cuda:0') 

cpu = torch.device('cpu')

N = 5000

# operations sur cpu
xc = torch.tensor(np.random.normal(size = (N,N)), device = cpu)
yc = torch.tensor(np.random.normal(size = (N,N)), device = cpu)

t0 = time.time()
zc = yc
for k in range(1000):
    zc = xc*zc
zc = torch.sum(zc)
print(zc)
print(time.time() - t0)
print(zc.device)

# operations sur gpu
xg = torch.tensor(np.random.normal(size = (N,N)), device = gpu)
yg = torch.tensor(np.random.normal(size = (N,N)), device = gpu)

t1 = time.time()
zg = yg
for k in range(1000):
    zg = xg*zg
zg = torch.sum(zg)
print(zg)
print(time.time() - t1)
print(zg.device)

print()