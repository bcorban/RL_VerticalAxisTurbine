import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import random
import sys
import torch
import math
sys.path.append('/home/adminit/RL_VerticalAxisTurbine/Fake_Carrousel/')
from surrogate_model.MLPmodel import MLP
m=2
tau=30
T=1127
mean=np.loadtxt("/home/adminit/RL_VerticalAxisTurbine/Fake_Carrousel/surrogate_model/NNet_files/means.txt") 
std=np.loadtxt("/home/adminit/RL_VerticalAxisTurbine/Fake_Carrousel/surrogate_model/NNet_files/stds.txt")
phase=np.loadtxt("/home/adminit/RL_VerticalAxisTurbine/Fake_Carrousel/data/phase.npy")

class CustomEnv(Env):
    def __init__(self,dict_env={}) -> None:
        super(CustomEnv, self).__init__()
        self.dict_env=dict_env
        self.action_space = Box(low=-0.55, high=0.55, shape=(1,))
        self.observation_space = Box(low=np.array([-1.73,-10, -50 , -10, -20,-10, -50 , -10, -20]), high=np.array([1.753,10, 10, 10, 20, 10, 10, 10, 20]))
        # self.state = np.array([phase[1],-1.44179,-1.67565, -0.950994, 1.0443, -1.4451,-2.6568, -1.79957, 3.21946],dtype='float32')
        self.state = np.array([-1.66706679, -0.36840033, -0.46035629, -0.99126627,  0.57170272, -0.26434656, -0.36760229, -1.73953709,  1.47209849],dtype='float32')
        # self.history=np.array([[phase[0], -1.4451,-2.6568, -1.79957, 3.21946], self.state],dtype='float32')
        self.surrogate_model= MLP(10,4,[128,128],mean,std, m,tau)
        self.surrogate_model.load_state_dict(torch.load("/home/adminit/RL_VerticalAxisTurbine/Fake_Carrousel/surrogate_model/NNet_files/trained_net.pt")) #loads trained MLP
        self.surrogate_model.eval()
        self.t=0
        self.reward=0
        self.bounds={"pitch":(-2.80,2.80),"Cp":(-7,7),"Cr":(-6,6),"Cm":(-15,16)}
    
        
    def step(self, action):
        input=self.state

        input=np.insert(input,5,action)
   
        partial_state=self.surrogate_model(torch.tensor(input.astype(float))).detach().numpy()

        self.state[0]=phase[self.t+1]
        self.state[5:]=self.state[1:5]
        self.state[1:5]=partial_state


        self.t+=1
        
        flags=[]

        if not (self.bounds["pitch"][0]<=self.state[1]<=self.bounds["pitch"][1]):
           flags.append("pitch")
        if not (self.bounds["Cp"][0]<=self.state[2]<=self.bounds["Cp"][1]):
           flags.append("Cp")
        if not (self.bounds["Cr"][0]<=self.state[3]<=self.bounds["Cr"][1]):
           flags.append("Cr")
        if not (self.bounds["Cm"][0]<=self.state[4]<=self.bounds["Cm"][1]):
           flags.append("Cm")
        
        if len(flags)>0:
            print(f"overshoot for {flags},t={self.t},state={self.state}")
            self.reward=0

        else:
            self.reward=(6+self.state[2])/6

        # if np.abs(self.state[2])<7 and not math.isnan(self.state[2]) and np.abs(self.state[1])<2.6:
        #     self.reward=(6+self.state[2])/13 #transform Cp so that the reward is always positive
        # else:
        #     self.reward=0
        #     print(f"overshoot,t={self.t}, action={action},state={self.state}")

        if self.t==2*int(T/tau):
            terminated = True
        else:
            terminated = False

        truncated =False
        info = {}
        
        # Return step information
        return self.state, self.reward, terminated, truncated, info
    

        
    def reset(self,seed=None,options=None):
        self.state = np.array([-1.66706679, -0.36840033, -0.46035629, -0.99126627,  0.57170272, -0.26434656, -0.36760229, -1.73953709,  1.47209849],dtype='float32')
        self.t=0
        self.reward=0
        info={}
        return self.state, info


    def render (self):
        print(f"t={self.t} -- state={self.state} -- reward={self.reward}")
