import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random
import sys
import torch
sys.path.append('/Users/baptiste/Desktop/EPFL/RL_VerticalAxisTurbine/')
from surrogate_model.MLPmodel import MLP
m=2
tau=10
mean=np.loadtxt("/Users/baptiste/Desktop/EPFL/RL_VerticalAxisTurbine/surrogate_model/NNet_files/means.txt") 
std=np.loadtxt("/Users/baptiste/Desktop/EPFL/RL_VerticalAxisTurbine/surrogate_model/NNet_files/stds.txt")
phase=np.loadtxt("/Users/baptiste/Desktop/EPFL/RL_VerticalAxisTurbine/data/phase.npy")

class CustomEnv(Env):
    def __init__(self):
        self.action_space = self.action_space = Box(low=0.014, high=0.11, shape=(1,))
        self.observation_space = Box(low=np.array([-2.5, -6 , -6, -15]), high=np.array([2.2, 7, 7, 3]))
        self.starting_states = np.loadtxt("../data/starting_history.npy")
        self.state = self.starting_states[-1]
        self.history=np.loadtxt("../data/starting_history.npy")
        self.surrogate_model= MLP(10,4,[64,128,64,32],mean,std, m,tau)
        self.t=0
        
      
    def step(self, action):
        partial_state=self.surrogate_model(torch.tensor(self.state[:5]+[action]+self.state[5:])).detach().numpy()
        self.state=np.array([phase[tau+self.t],partial_state,self.history[self.t,5:]])
        np.append(self.history,self.state)
        self.t+=1
        reward=self.state[2]
        
        if self.t==200: 
            done = True
        else:
            done = False
        
        info = {}
        
        # Return step information
        return self.state, reward, done, info
    
    
    def reset(self):
        self.state = self.starting_states[-1]
        self.history=np.loadtxt("../data/starting_history.npy")
        self.t=0
        return self.state
    def render (self, mode="human"):
        s = "state: {:2d}  reward: {:2d}  info: {}"
        print(s.format(self.state[:5], self.reward, self.info))