# import gymnasium as gym
# import torch
# import numpy as np
import ray
# import RL_
# import matplotlib.pyplot as plt
# from RL_.envs.CustomEnv import CustomEnv

# from ray.rllib.algorithms.algorithm import Algorithm

from ray.rllib.algorithms.sac import SAC,SACConfig
# from custom_callbacks import CustomCallbacks
from config import CONFIG_SAC


def RL_loop():

    algo = SAC(config=CONFIG_SAC)

    for epoch in range(CONFIG_SAC["N_epoch"]):
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
