import matlab.engine
import time
eng = matlab.engine.start_matlab()

path = "/Users/PIVUSER/Documents/MATLAB"

eng.addpath(path, nargout= 0 )

path = "/Users/PIVUSER/Desktop/RL_VerticalAxisTurbine/Carousel"

eng.addpath(path, nargout= 0 )

eng.start_lc(nargout=0)
time.sleep(2)
eng.stop_lc(nargout=0)
d=eng.workspace['d']
print(d)

