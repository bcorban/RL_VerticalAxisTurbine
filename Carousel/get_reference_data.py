# this script aims at performing a few non actuated episodes to get reference loads 

from param_matlab import param, m, NI
import gclib
from train import RL_loop
from setup_galil import setup_g
from RL_.envs.CustomEnv import CustomEnv
import gymnasium as gym
import RL_
import matlab.engine
g = gclib.py()

# -----------------Connect to galil and set parameters ----------------------


eng = matlab.engine.start_matlab()
path = "/Users/PIVUSER/Desktop/RL_VerticalAxisTurbine/Carousel"
eng.addpath(path, nargout=0)
path = "/Users/PIVUSER/Documents/MATLAB"
eng.addpath(path, nargout=0)
g = gclib.py()
c = g.GCommand
g.GOpen("192.168.255.200 --direct -s ALL")
print(g.GInfo())

env = gym.make('CustomEnv')

env.reset()
while env.history_phase_cont[env.i]// 360<30:
    env.read_state()
env.save_data()
env.stop_E()

g.GClose()
