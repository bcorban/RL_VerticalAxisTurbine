import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random

class CustomEnv(Env):
    def __init__(self):
        self.action_space = self.action_space = Box(low=0.014, high=0.11, shape=(1,))
        self.observation_space = Box(low=np.array([-2.5, -6 , -6, -15]), high=np.array([2.2, 7, 7, 3]))
        self.history = np.loadtxt("../data/starting_history.npy")
        
    def step(self, action):
        self.state += action -1 
        self.shower_length -= 1 
        
        if self.state >=37 and self.state <=39: 
            reward =1 
        else: 
            reward = -1 
        
        if self.shower_length <= 0: 
            done = True
        else:
            done = False
        
        info = {}
        
        # Return step information
        return self.state, reward, done, info
    
    def reset(self):
        self.state = 38 + random.randint(-3,3)
        self.shower_length = 60 
        return self.state