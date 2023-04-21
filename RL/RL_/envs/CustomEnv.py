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
tau=30
T=1127
mean=np.loadtxt("/home/adminit/RL_VerticalAxisTurbine/surrogate_model/NNet_files/means.txt") 
std=np.loadtxt("/home/adminit/RL_VerticalAxisTurbine/surrogate_model/NNet_files/stds.txt")
phase=np.loadtxt("/home/adminit/RL_VerticalAxisTurbine/data/phase.npy")

class CustomEnv(Env):
    def __init__(self,dict_env={}) -> None:
        super(CustomEnv, self).__init__()
        self.dict_env=dict_env
        self.action_space = self.action_space = Box(low=-0.3, high=0.3, shape=(1,))
        self.observation_space = Box(low=np.array([-1.73,-10, -50 , -6, -15,-10, -50 , -6, -15]), high=np.array([1.753,5, 7, 6, 16,5, 7, 6, 16]))
        # self.state = np.array([phase[1],-1.44179,-1.67565, -0.950994, 1.0443, -1.4451,-2.6568, -1.79957, 3.21946],dtype='float32')
        self.state = np.array([-1.66706679, -0.36840033, -0.46035629, -0.99126627,  0.57170272, -0.26434656, -0.36760229, -1.73953709,  1.47209849],dtype='float32')
        # self.history=np.array([[phase[0], -1.4451,-2.6568, -1.79957, 3.21946], self.state],dtype='float32')
        self.surrogate_model= MLP(10,4,[128,128],mean,std, m,tau)
        self.surrogate_model.load_state_dict(torch.load("/home/adminit/RL_VerticalAxisTurbine/surrogate_model/NNet_files/trained_net.pt")) #loads trained MLP
        self.surrogate_model.eval()
        self.t=0
        
      
    def step(self, action):
        input=self.state

        input=np.insert(input,5,action)
   
        # print(f"state t={self.t}: {self.state}")
        # input=np.zeros(10)
        # input[:5]=self.state
        # input[5]=action
        # input[6:]=self.history[-2,1:]
        # print(f"surrogate model input : {input}")
#         input=np.array([-1.57465772, -1.44179798, -1.67565033,-0.95099476,
#    1.044348,  0.156754614, -1.44507792, -2.65680786,
#   -1.79957228,  3.21946533])
        partial_state=self.surrogate_model(torch.tensor(input.astype(float))).detach().numpy()
        # self.state=np.concatenate((np.array([phase[tau+self.t]]),partial_state,self.history[self.t,1:5]))
        # print(partial_state)
        self.state[0]=phase[self.t+1]
        self.state[5:]=self.state[1:5]
        self.state[1:5]=partial_state
        # print(f"following state t={self.t}: {self.state}")
        # self.history=np.vstack((self.history,self.state))

        self.t+=1
        # print(self.t)
        if np.abs(self.state[2])<7 and not math.isnan(self.state[2]) and np.abs(self.state[1])<2.6:
            reward=(6+self.state[2])/13 #transform Cp so that the reward is always positive
        else:
            reward=-5
            print(f"overshoot,t={self.t}, action={action},state={self.state}")

        if self.t==1*int(T/tau):
            terminated = True
        else:
            terminated = False
        truncated =False
        info = {}
        
        # Return step information
        return self.state, reward, terminated, truncated, info
    
    
    def reset(self,seed=None,options=None):
        # print("reset")
        self.state = np.array([-1.66706679, -0.36840033, -0.46035629, -0.99126627,  0.57170272, -0.26434656, -0.36760229, -1.73953709,  1.47209849],dtype='float32')
        # self.state = np.array([phase[1],-1.44179,-1.67565, -0.950994, 1.0443, -1.4451,-2.6568, -1.79957, 3.21946],dtype='float32')
        # self.history=np.array([[phase[0], -1.4451,-2.6568, -1.79957, 3.21946], self.state],dtype='float32')
        self.t=0
        info={}
        return self.state, info


    def render (self, mode="human"):
        s = "state: {:2d}  reward: {:2d}  info: {}"
        print(s.format(self.state, self.reward, self.info))