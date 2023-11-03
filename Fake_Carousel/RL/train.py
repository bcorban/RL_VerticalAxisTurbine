import gymnasium as gym
import torch
import numpy as np
import ray
import RL_
import matplotlib.pyplot as plt
from RL_.envs.CustomEnv import CustomEnv,phase,mean,std
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.sac import SAC,SACConfig
from custom_callbacks import CustomCallbacks
from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer

tau=30
T=1127
params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "text.usetex" : True,
          "font.family" : "serif",
          "font.serif" : ["Times"],
          "font.size":15}
plt.rcParams.update(params)



# class CustomReplayBuffer(ReplayBuffer):
#     def add(self, data):
#         # Check the condition based on the state information
#         if self.should_store_transition(data):
            
#             print(data.__getitem__('default_policy').__getitem__('obs'))
#             print(data.__getitem__('default_policy').__getitem__('t'))
#             print(data.__getitem__('default_policy').__getitem__('infos')[0]['transient'])
#             super().add(data)

#     def should_store_transition(self, data):
#         # Implement your condition here based on the state information
#         # For example, if the position is in the state, you can check if it meets your criteria
#         # return data.__getitem__('default_policy').__getitem__('obs')[0][0]>360*3
#         return True

CONFIG_PPO = {
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
    "lr": 0.5e-4,
    "lr_schedule": None,
    "vf_loss_coeff": 1.0,
    "model": {
        "fcnet_hiddens":[36,36],
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

CONFIG_SAC={
#COMMON config
        "log_level": "INFO",
		"env": CustomEnv,
		# "env_config": ENV_CONFIG, #the env config is the dictionary that's pass to the environment when built
		"num_gpus": 0,
		"num_workers": 0, # int(ressources['CPU'])
		"explore": True,
		"exploration_config": {
			"type": "StochasticSampling",
		},
#   "replay_buffer_config": {"type" :CustomReplayBuffer,},
#   "replay_buffer_config": {"type" :CustomReplayBuffer,},
    "framework": "torch",
    "callbacks":CustomCallbacks,
     "prioritized_replay": True,
     "gamma": 1
}



train=True
ALGO=["SAC","PPO"][0] 

ray.shutdown() #shutdown before re-init
ray.init() #re-init
if ALGO=="SAC":
    algo = SAC(config=CONFIG_SAC)
elif ALGO=="PPO":
    algo = PPO(config=CONFIG_PPO)

if train:
    for epoch in range(150):
        result=algo.train()
        # print(result)
        print('epoch : ',epoch)
        # print(pretty_print(result))
        # if epoch%1000==0:
        #     checkpoint_dir = algo.save()
        #     print(f"Checkpoint saved in directory {checkpoint_dir}")
    checkpoint_dir = algo.save() #save the model 
    print(f"Checkpoint saved in directory {checkpoint_dir}") 
    ray.shutdown()
else:
    checkpoint_dir="/home/adminit/ray_results/SAC_CustomEnv_2023-04-26_16-51-24e08kmjvx/checkpoint_000150"
    ray.shutdown()

algo = Algorithm.from_checkpoint(checkpoint_dir) #load the state of the algorithm where it was : Optimizer state, weights, ...
# policy=algo.get_policy() #get the policy 
algo.get_policy().config["explore"] = False
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
    print(a)
    pitch_from_action_list.append(pitch_from_action_list[-1]+a[0])
    s,r,d,t,i= env.step(a[0])
    hist=np.vstack((hist,s))
    episode_reward += r
    env.render()

env.close()
print(f"\n episode reward: {episode_reward} -- Cp_mean={(6*episode_reward/int(T/tau)-6)*std[2]+mean[2]}\n")

print(len(hist[:,0]))
plt.figure()
plt.ylabel("$\\alpha$")
plt.plot(np.array(list(range(len(hist[:,0]))))/int(T/tau),hist[:,1]*std[1]+mean[1],'o-',color='black')
plt.plot((np.array(list(range(len(hist[:,0]))))/int(T/tau))[:],np.array(pitch_from_action_list)*std[1]+mean[1],'-',color='black',alpha=0.5)
plt.xlabel("$t/T$")
plt.grid()
plt.savefig('1.eps')

plt.figure()
plt.ylabel("$C_p$")
plt.plot(np.array(list(range(len(hist[:,0]))))/int(T/tau),hist[:,2]*std[2]+mean[2],'o-',color='black')
plt.xlabel("$t/T$")
plt.grid()
plt.savefig('2.eps')

plt.figure()
plt.title("phase")
plt.plot(np.array(list(range(len(hist[:,0]))))/int(T/tau),hist[:,0]*std[0]+mean[0],'o-')
plt.xlabel("$t/T$")
plt.show()
plt.savefig('3.eps')