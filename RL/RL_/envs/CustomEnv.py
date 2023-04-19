import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import random
import sys
import torch
import math
sys.path.append('/home/adminit/RL_VerticalAxisTurbine/')
from surrogate_model.MLPmodel import MLP
m=2
tau=10
mean=np.loadtxt("/home/adminit/RL_VerticalAxisTurbine/surrogate_model/NNet_files/means.txt") 
std=np.loadtxt("/home/adminit/RL_VerticalAxisTurbine/surrogate_model/NNet_files/stds.txt")
phase=np.loadtxt("/home/adminit/RL_VerticalAxisTurbine/data/phase.npy")

class CustomEnv(Env):
    def __init__(self,dict_env={}) -> None:
        super(CustomEnv, self).__init__()
        self.dict_env=dict_env
        self.action_space = self.action_space = Box(low=-0.06, high=0.071, shape=(1,))
        self.observation_space = Box(low=np.array([-2,-2.5, -6 , -6, -15]), high=np.array([2,2.2, 7, 7, 15]))
        self.starting_states = np.loadtxt("../data/starting_history.npy").astype('float32')
        self.state = self.starting_states[-1]
        self.history=np.loadtxt("../data/starting_history.npy")
        print(self.history)
        self.surrogate_model= MLP(10,4,[64,128,64,32],mean,std, m,tau)
        self.t=0
        
      
    def step(self, action):
        # input=self.state
        # print(input)
        # input=np.insert(input,5,action)
        # print(action)
        # print(input)
        print(f"state t={self.t}: {self.state}")
        input=np.zeros(10)
        input[:5]=self.state
        input[5]=action
        input[6:]=self.history[self.t,1:]
        print(f"surrogate model input : {input}")
        
        partial_state=self.surrogate_model(torch.tensor(input.astype(float))).detach().numpy()
        # self.state=np.concatenate((np.array([phase[tau+self.t]]),partial_state,self.history[self.t,1:5]))
        
        self.state[0]=phase[tau+self.t]
        self.state[1:]=partial_state
        print(f"following state t={self.t}: {self.state}")
        self.history=np.vstack((self.history,self.state))
        # print(np.shape(self.history))
        self.t+=1

        if np.abs(self.state[2])<7 and not math.isnan(self.state[2]):
            reward=self.state[2]/225
        else:
            reward=-10
            print("overshoot")

        if self.t==5:
            # print("terminated")
            terminated = True
        else:
            terminated = False
        truncated =False
        info = {}
        
        # Return step information
        return self.state, reward, terminated, truncated, info
    
    
    def reset(self,seed=None,options=None):
        # print("reset")
        self.state = self.starting_states[-1]
        
        self.history=np.loadtxt("../data/starting_history.npy")
        self.t=0
        info={}
        return self.state, info
    def render (self, mode="human"):
        s = "state: {:2d}  reward: {:2d}  info: {}"
        print(s.format(self.state, self.reward, self.info))