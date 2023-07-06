import numpy as np
import gclib 

# import matlab.engine
from load_mat import get_param

param=get_param()
param['R4']=np.array(param['R4'])[:,[0,1,4]]
m=param['m']
NI=param['NI']


