import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import random
import sys
import torch
import math
import gclib
import matlab.engine
import time
sys.path.append("/Users/PIVUSER/Desktop/tmp_baptiste/Carousel/RL_/envs")

from param_matlab import param,m,NI

eng = matlab.engine.start_matlab()
path = "/Users/PIVUSER/Desktop/tmp_baptiste/Carousel/RL_/envs"
eng.addpath (path, nargout= 0 )
path = "/Users/PIVUSER/Documents/MATLAB"
eng.addpath(path, nargout= 0 )
g=gclib.py()
c=g.GCommand
g.GOpen('192.168.255.200 --direct -s ALL')
print(g.GInfo())

class CustomEnv(Env):
    
    def __init__(self,dict_env={}) -> None:
        super(CustomEnv, self).__init__()
        # self.dict_env=dict_env
        self.action_space = Box(low=-0.55, high=0.55, shape=(1,))
        self.observation_space = Box(low=np.array([-2,-2,-2]), high=np.array([2,2,2]))       

        self.start_E()
        self.wait_N_rot(3)        
        self.N_ep_without_homing=300
        self.N_transient_effects=3
        self.N_max_ep=10

        self.episode_counter=0

    def step(self, action):
        self.pitch(action)
        self.i+=1
        self.state=self.read_state()
        self.history_states[self.t]=self.state
        
        nrot = self.state['phase_index'] // 360 #change index to get phase
        
        if nrot <= self.N_transient_effects + self.n_rot_ini:
            self.reward=0
        else:
            self.reward=self.state['Cp']/value +value
        if nrot>=self.N_max_ep:
            terminated = True
            self.pitch(0) #pitch back to 0 at the end of an episode
        else:
            terminated = False

        truncated =False
        info = {}
        
        # Return step information
        return self.state, self.reward, terminated, truncated, info
    

        
    def reset(self,seed=None,options=None):
        self.episode_counter+=1
        self.history_states=np.zeros(10000,4)
        self.history_volts=np.zeros(10000,8)
        self.history_volts_raw=np.zeros(10000,8)
        self.i=0
        
        if self.episode_counter!=0:
            eng.stop_lc(nargout=0)
            self.save_data()
            
        if self.episode_counter!=0 and self.episode_counter%self.N_ep_without_homing==0:
            print("homing")
            self.stop_E()
            self.home()
            self.get_offset()
            self.start_E()
            self.wait_N_rot(3)
        
        #-----------------------------------
        eng.start_lc(nargout=0) #Start the loadcell
        self.t_start=time.time()
        self.n_rot_ini=c('MG _TPE')/ m[1]['es']//360
        self.state = self.read_state()
        self.reward=0
        info={}

        # Wait 
        return self.state, info


    def render(self):
        print(f"t={self.t} -- state={self.state} -- reward={self.reward}")
    
    def home():
        print('homing...')
        eng.my_quick_home(nargout=0)
        

    def pitch(action_abs):

        print('pitching')
        g.GCommand(f"PAF={action_abs}")


    def read_state(self):
        c=g.GCommand
        print('state')
        galil_output=c('MG @AN[1],@AN[2],@AN[3],@AN[5],@AN[7],_TPE,_TDF,_TPF')
        galil_output[5]-=self.n_rot_ini*360*m[1]['es']
        volts_raw=galil_output[0:5]
        volts=-(volts_raw)-self.offset
        phase= galil_output[5] / m[1]['es'] % 360# In degrees
        phase_cont = galil_output[5] / m[1]['es']  #In degrees
        self.history_volts_raw[self.i]=volts_raw #check formats !!!
        self.history_volts[self.i]=volts #check formats !!!
        
        
        
        
        

        return np.zeros(3)

    def start_E():
        print("Starting motor E")
        g.GCommand('SHE')
        g.GCommand(f"JGE={param['JG']}")
        g.GCommand('BGE')

        #initialize position tracking
        g.GCommand("PTF=1")

    def stop_E():
        print("Stopping motor E")
        g.GCommand("ST")

    def wait_N_rot(self,N):
        print(f"waiting for {N} periods")
        phase_ini=self.read_state(g)['phase_index']//360
        phase=phase_ini
        while phase-phase_ini< N:
            phase=self.read_state(g)['phase_index']//360
            
    def save_data(self):
        print("saving data")
        
    def get_offset(self):
        self.offset=value
        
    def close(self):
        g.GClose()
        
    def fill_outliers(self):
        clean_list=[]
        return clean_list