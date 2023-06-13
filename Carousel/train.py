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
from replay_buffer import SB3_CustomReplayBuffer

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
    Nrot_ep=1
    Nts_ep=83
    env=gym.make('RL_/CustomEnv-v0')
    model = SAC("MlpPolicy", env, verbose=1,replay_buffer_class=SB3_CustomReplayBuffer,learning_rate=3e-4,buffer_size=500000,batch_size=512,ent_coef=0.1,train_freq=Nrot_ep*Nts_ep)
    model.learn(total_timesteps=10000, progress_bar=True,train_freq=10)