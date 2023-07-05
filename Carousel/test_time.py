import time 
import matplotlib.pyplot as plt
import numpy as np
import win_precise_time as wpt

N=5000
t1=np.empty((N,))
t2=np.empty((N,))
now=wpt.time()
for n in range(N):
    t1[n]=wpt.time()-now
    wpt.sleep(0.002)
    t2[n]=time.time_ns()/ (10 ** 9)-now

plt.figure()
plt.plot(t1,'-o')
plt.plot(t2,'-o')
plt.show()