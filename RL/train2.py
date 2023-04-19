import gymnasium as gym
import torch
import numpy as np
import ray
import RL_
import matplotlib.pyplot as plt
from RL_.envs.CustomEnv import CustomEnv
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPO

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
    "gamma": 0.99999,
    "use_critic": True,
    "use_gae": True, #Generalized Advantage Estimate
    "lambda": 1,
    "kl_coeff": 0.2,
    "rollout_fragment_length":256, #number of steps in the environment for each Rollout Worker
    "train_batch_size": 1024, 
    "sgd_minibatch_size": 64,
    "shuffle_sequences": True, #Kind of experience replay for PPO
    "num_sgd_iter": 16,
    "lr": 5e-4,
    "lr_schedule": None,
    "vf_loss_coeff": 1.0,
    "model": {
        "fcnet_hiddens":[36,36],
        "fcnet_activation":"relu",
        "vf_share_layers": False, 
    },
    "entropy_coeff": 0.0,
    "entropy_coeff_schedule": None,
    "clip_param": 0.3,
    "vf_clip_param": 20.0,
    "grad_clip": None,
    "observation_filter": "NoFilter"
	}


train=True

ray.shutdown() #shutdown before re-init
ray.init() #re-init
algo = PPO(config=CONFIG)

if train:
    for epoch in range(100):
        result=algo.train()
        print('epoch : ',epoch)
        # print(pretty_print(result))

    checkpoint_dir = algo.save() #save the model 
    print(f"Checkpoint saved in directory {checkpoint_dir}") 
    ray.shutdown()
else:
    checkpoint_dir="/home/adminit/ray_results/PPO_CustomEnv_2023-04-18_10-03-42mtpy_1lj/checkpoint_000100"
    ray.shutdown()

algo = Algorithm.from_checkpoint(checkpoint_dir) #load the state of the algorithm where it was : Optimizer state, weights, ...
policy=algo.get_policy() #get the policy 

env = gym.make('CustomEnv')
episode_reward = 0
d = False
s,i = env.reset()
hist=np.loadtxt("../data/starting_history.npy")

b=0
while not d:
    
    # logits,_= policy.model({'obs':torch.from_numpy( np.expand_dims(s,axis=0).astype(float))})
    # # help(policy.model)
    # # print(policy.model({'obs':torch.from_numpy( np.expand_dims(s,axis=0).astype(float))}))
    # a=np.argmax(logits.detach().numpy())
    # s,r,d,t,i= env.step(a)
    # print(f"timestep : {b}, action : {a}")
    # b+=1

    # hist=np.vstack((hist,s))

    a= policy.compute_single_action(s)
    print(f"timestep : {b}, action : {a[0]}")
    s,r,d,t,i= env.step(a[0])
    hist=np.vstack((hist,s))
    episode_reward += r
    b+=1

env.close()
print(f"\n episode reward: {episode_reward} \n")

print(len(hist[:,0]))
plt.figure()
plt.plot(np.array(list(range(len(hist[:,0]))))/225,hist[:,1])
plt.figure()
plt.plot(np.array(list(range(len(hist[:,0]))))/225,hist[:,2])
plt.figure()
plt.plot(np.array(list(range(len(hist[:,0]))))/225,hist[:,0])

plt.show()
