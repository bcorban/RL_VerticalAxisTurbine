# import gymnasium as gym
# import torch
# import numpy as np
import ray

# import RL_
# import matplotlib.pyplot as plt
# from RL_.envs.CustomEnv import CustomEnv

# from ray.rllib.algorithms.algorithm import Algorithm

from ray.rllib.algorithms.sac import SAC, SACConfig
import RL_
# from custom_callbacks import CustomCallbacks
from config_SAC import CONFIG_SAC, CONFIG_TRAIN
import gym
import ray
from stable_baselines3 import SAC

# ray.rllib.utils.check_env([gym.make('CustomEnv')])
def RL_loop_rllib():
    algo = SAC(config=CONFIG_SAC)

    for epoch in range(CONFIG_TRAIN["N_epoch"]): #training loop
        result = algo.train()
        print(result)
        print("epoch : ", epoch)
        # print(pretty_print(result))
        # if epoch%1000==0:
        #     checkpoint_dir = algo.save()
        #     print(f"Checkpoint saved in directory {checkpoint_dir}")

    checkpoint_dir = algo.save()  # save the model
    print(f"Checkpoint saved in directory {checkpoint_dir}")
    ray.shutdown()

def RL_loop_sb3():
    env=gym.make('RL_/CustomEnv-v0')
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, progress_bar=True)