import gymnasium as gym
import torch
import numpy as np
import ray
import RL_
import matplotlib.pyplot as plt
from RL_.envs.CustomEnv import CustomEnv

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer
from ray.rllib.algorithms.sac import SAC,SACConfig
from custom_callbacks import CustomCallbacks

def RL_loop(c):
    params = {"ytick.color" : "black",
            "xtick.color" : "black",
            "axes.labelcolor" : "black",
            "axes.edgecolor" : "black",
            "text.usetex" : True,
            "font.family" : "serif",
            "font.serif" : ["Times"],
            "font.size":15}
    plt.rcParams.update(params)
    
    

    class CustomReplayBuffer(ReplayBuffer):
        def add(self, data):
            # Check the condition based on the state information
            if self.should_store_transition(data):
                
                print(data.__getitem__('default_policy').__getitem__('obs')[0][0])
                super().add(data)

        def should_store_transition(self, data):
            # Implement your condition here based on the state information
            # For example, if the position is in the state, you can check if it meets your criteria
            return data.__getitem__('default_policy').__getitem__('obs')[0][0]>360*3
        
        
        
    CONFIG_SAC={
    #COMMON config
            "log_level": "INFO",
            "env": CustomEnv,
            # "env_config": ENV_CONFIG, #the env config is the dictionary that's pass to the environment when built
            "num_gpus": 0,
            "num_workers": 4, # int(ressources['CPU'])
            "explore": True,
            "exploration_config": {
                "type": "StochasticSampling",
            },
        "framework": "torch",
        "replay_buffer_config": {"type" :CustomReplayBuffer,},
        # "callbacks":CustomCallbacks,
        "prioritized_replay": True,
        "gamma": 1
    }
    

    algo = SAC(config=CONFIG_SAC)



    for epoch in range(150):
        result=algo.train()
        print(result)
        print('epoch : ',epoch)
        # print(pretty_print(result))
        # if epoch%1000==0:
        #     checkpoint_dir = algo.save()
        #     print(f"Checkpoint saved in directory {checkpoint_dir}")
    checkpoint_dir = algo.save() #save the model 
    print(f"Checkpoint saved in directory {checkpoint_dir}") 
    ray.shutdown()
