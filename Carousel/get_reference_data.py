# this script aims at performing a few non actuated episodes to get reference loads 

from param_matlab import param, m, NI
import gclib
from train import RL_loop
from setup_galil import setup_g
from RL_.envs.CustomEnv import CustomEnv
import gymnasium as gym
import RL_
import matlab.engine

eng = matlab.engine.start_matlab()
path = "/Users/PIVUSER/Desktop/RL_VerticalAxisTurbine/Carousel"
eng.addpath(path, nargout=0)
path = "/Users/PIVUSER/Documents/MATLAB"
eng.addpath(path, nargout=0)

g = gclib.py()
c = g.GCommand
g.GOpen("192.168.255.200 --direct -s ALL")
print(g.GInfo())

setup_g(g)

env = gym.make('CustomEnv')
env.reset()
env.wait_N_rot(40)
env.close()

g.GClose()
