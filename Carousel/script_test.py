import matlab.engine
import time
import numpy as np

# eng = matlab.engine.start_matlab()
eng = matlab.engine.connect_matlab()
eng.addpath(r'/home/adminit/MATLAB_scripts',nargout=0)
s=time.time()
eng.param_definition(nargout=0)

para=eng.workspace['CDnaca0']
print(para)
print(f'import time : {time.time()-s}')

# a=5
# s=time.time()
# for i in range(100000000):
#     a=np.cos(a)**5
# print(f'python loop execution time : {time.time()-s}')


# eng = matlab.engine.connect_matlab()
# eng.addpath(r'/home/adminit/MATLAB_scripts',nargout=0)
# s=time.time()
# a = eng.test()
# print(a)
# print(f'python loop execution time : {time.time()-s}')