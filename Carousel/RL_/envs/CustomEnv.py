import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import random
import sys
import torch
import math
import gclib
import matlab.engine
sys.path.append("/Users/PIVUSER/Desktop/tmp_baptiste/Carousel/RL_/envs")
from Galil_functions import *
from param_matlab import param,m,NI

eng = matlab.engine.start_matlab()
path = "/Users/PIVUSER/Desktop/tmp_baptiste/Carousel/RL_/envs"
eng.addpath (path, nargout= 0 )
g=gclib.py()
c=g.GCommand
g.GOpen('192.168.255.200 --direct -s ALL')
print(g.GInfo())

class CustomEnv(Env):
    def __init__(self,dict_env={}) -> None:
        super(CustomEnv, self).__init__()
        self.dict_env=dict_env
        self.action_space = Box(low=-0.55, high=0.55, shape=(1,))
        self.observation_space = Box(low=np.array([-2,-2,-2]), high=np.array([2,2,2]))       
        self.state = read_state(g)
        self.begin=True
        self.reward=0
        start_E(g)
        wait_N_rot(g,3)
        self.state=read_state(g)
        self.episode_counter=0
        self.N_ep_without_homing=300
        self.N_transient_effects=3
        self.N_max_ep=10
        self.history=np.zeros(10000,4)
        self.t=0


    def step(self, action):
        pitch(action,g)
        self.state=read_state(g)
        self.history[self.t]=self.state
        self.t+=1
        nrot = self.state['phase_index'] // 360 #change index to get phase
        
        if nrot <= self.N_transient_effects + self.n_rot_ini:
            self.reward=0
        else:
            self.reward=self.state['Cp']/value +value
        if nrot>=self.N_max_ep:
            terminated = True
        else:
            terminated = False

        truncated =False
        info = {}
        
        # Return step information
        return self.state, self.reward, terminated, truncated, info
    

        
    def reset(self,seed=None,options=None):
        self.episode_counter+=1
        if self.episode_counter%self.N_ep_without_homing==0:
            print("homing")
            stop_E()
            home()
            start_E()
        if self.episode_counter!=0:
            save_data()
        
        self.state = read_state(g)
        self.n_rot_ini=self.state['phase_index'] //360 #phase
        self.reward=0
        self.history=np.zeros(10000,4)
        self.t=0
        info={}

        # Wait 
        return self.state, info


    def render (self):
        print(f"t={self.t} -- state={self.state} -- reward={self.reward}")