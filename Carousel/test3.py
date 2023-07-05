
import matlab.engine
import time
eng = matlab.engine.start_matlab()
path = "/Users/PIVUSER/Desktop/RL_VerticalAxisTurbine/Carousel"
eng.addpath(path, nargout=0)
path = "/Users/PIVUSER/Documents/MATLAB"
eng.addpath(path, nargout=0)
path = "/Users/PIVUSER/Desktop/RL_VerticalAxisTurbine/Carousel/RL_/envs"
eng.addpath(path, nargout=0)
# eng.desktop(nargout=0)

eng.start_lc(nargout=0)  # Start the loadcell
# t_start.value = wpt.time()
# time.sleep(15)
# t_start=eng.worskpace["a"]
t_start=eng.workspace["t_start"]
print(t_start)
# eng.stop_lc(nargout=0)