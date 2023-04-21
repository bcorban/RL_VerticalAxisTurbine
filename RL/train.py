import gymnasium as gym
import torch
import numpy as np
import ray
import RL_
import matplotlib.pyplot as plt
from RL_.envs.CustomEnv import CustomEnv,phase
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPO

tau=30
T=1127

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
    "gamma": 1,
    "use_critic": True,
    "use_gae": True, #Generalized Advantage Estimate
    "lambda": 1,
    "kl_coeff": 0.2,
    "rollout_fragment_length":256, #number of steps in the environment for each Rollout Worker
    "train_batch_size": 1024, 
    "sgd_minibatch_size": 64,
    "shuffle_sequences": True, #Kind of experience replay for PPO
    "num_sgd_iter": 16,
    "lr": 1e-4,
    "lr_schedule": None,
    "vf_loss_coeff": 1.0,
    "model": {
        "fcnet_hiddens":[128,128],
        "fcnet_activation":"relu",
        "vf_share_layers": False, 
    },
    "entropy_coeff": 0.0,
    "entropy_coeff_schedule": None,
    "clip_param": 0.1,
    "vf_clip_param": 10.0,
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
    checkpoint_dir="/home/adminit/ray_results/PPO_CustomEnv_2023-04-21_11-43-05bgfkn3fq/checkpoint_000100"
    ray.shutdown()

algo = Algorithm.from_checkpoint(checkpoint_dir) #load the state of the algorithm where it was : Optimizer state, weights, ...
# policy=algo.get_policy() #get the policy 

env = gym.make('CustomEnv')
episode_reward = 0
d = False
s,i = env.reset()
hist=np.array([-1.66706679, -0.36840033, -0.46035629, -0.99126627,  0.57170272, -0.26434656, -0.36760229, -1.73953709,  1.47209849],dtype='float32')
pitch_from_action_list=[-0.36840033]

while not d:
    
    # logits,_= policy.model({'obs':torch.from_numpy( np.expand_dims(s,axis=0).astype(float))})
    # # help(policy.model)
    # # print(policy.model({'obs':torch.from_numpy( np.expand_dims(s,axis=0).astype(float))}))
    # a=np.argmax(logits.detach().numpy())
    # s,r,d,t,i= env.step(a)
    # print(f"timestep : {b}, action : {a}")
    # b+=1

    # hist=np.vstack((hist,s))

    a= algo.compute_single_action(s)
    pitch_from_action_list.append(pitch_from_action_list[-1]+a[0])
    s,r,d,t,i= env.step(a[0])
    hist=np.vstack((hist,s))
    episode_reward += r
    env.render()

env.close()
print(f"\n episode reward: {episode_reward} -- Cp_mean={(episode_reward*13)/int(T/tau)-6}\n")

print(len(hist[:,0]))
plt.figure()
plt.title("pitch")
plt.plot(np.array(list(range(len(hist[:,0]))))/int(T/tau),hist[:,1],'o-')
plt.plot((np.array(list(range(len(hist[:,0]))))/int(T/tau))[:],pitch_from_action_list,'-')
plt.figure()
plt.title("Cp")
plt.plot(np.array(list(range(len(hist[:,0]))))/int(T/tau),hist[:,2],'o-')
plt.figure()
plt.title("phase")
plt.plot(np.array(list(range(len(hist[:,0]))))/int(T/tau),hist[:,0],'o-')

plt.show()
