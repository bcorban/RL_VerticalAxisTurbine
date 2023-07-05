import numpy as np
import matplotlib.pyplot as plt


t_a=1.005
d=np.load('test.npz')
tpf=d['tpf']
t=d['t']
plt.figure()
plt.plot(t,tpf, '-o')
plt.scatter([t_a],[6])
plt.show()