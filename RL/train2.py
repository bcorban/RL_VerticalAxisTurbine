import gymnasium as gym
import torch
import numpy as np
import ray
import RL_
import matplotlib.pyplot as plt
from RL_.envs.CustomEnv import CustomEnv

from ray.rllib.algorithms.algorithm import Algorithm
CONFIG = {
    #COMMON config
		"env": CustomEnv,
		# "env_config": ENV_CONFIG, #the env config is the dictionary that's pass to the environment when built
		"num_gpus": 0,
		"num_workers": 4, # int(ressources['CPU'])
		"explore": True,
		"exploration_config": {
			"type": "StochasticSampling",
		},
    "framework": "torch", #I prefer tensorflow but feel free to use pytorch
    # PPO config
    "gamma": 0.95,
    "use_critic": True,
    "use_gae": True, #Generalized Advantage Estimate
    "lambda": 1,
    "kl_coeff": 0.2,
    "rollout_fragment_length":256, #number of steps in the environment for each Rollout Worker
    "train_batch_size": 1024, 
    "sgd_minibatch_size": 64,
    "shuffle_sequences": True, #Kind of experience replay for PPO
    "num_sgd_iter": 16,
    "lr": 1e-3,
    "lr_schedule": None,
    "vf_loss_coeff": 1.0,
    "model": {
        "vf_share_layers": False, 
    },
    "entropy_coeff": 0.0,
    "entropy_coeff_schedule": None,
    "clip_param": 0.4,
    "vf_clip_param": 10.0,
    "grad_clip": None,
    "observation_filter": "NoFilter"
	}



from ray.rllib.algorithms.ppo import PPO
ray.shutdown() #shutdown before re-init
ray.init() #re-init
algo = PPO(config=CONFIG)
for epoch in range(100):
	result=algo.train()
	print('epoch : ',epoch)

checkpoint_dir = algo.save() #save the model 
print(f"Checkpoint saved in directory {checkpoint_dir}") 
ray.shutdown()

# checkpoint_dir="/home/adminit/ray_results/PPO_CustomEnv_2023-04-14_16-04-09uf83tjui/checkpoint_001000"
algo = Algorithm.from_checkpoint(checkpoint_dir) #load the state of the algorithm where it was : Optimizer state, weights, ...
policy=algo.get_policy() #get the policy 

env = gym.make('CustomEnv')
episode_reward = 0
d = False
s,i = env.reset()
h=[s]
while not d:
    a= policy.compute_single_action(s)
    print(a)
    s,r,d,t,i= env.step(a[0])
    episode_reward += r
    h.append(s)
env.close()
h=np.array(h)
plt.figure()
plt.plot(list(range(len(h[:,0]))),h[:,1])
plt.figure()
plt.plot(list(range(len(h[:,0]))),h[:,2])

plt.show()
