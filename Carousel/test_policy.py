import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import random
import time
from distutils.util import strtobool
import gclib
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import RL_ 
import win_precise_time as wpt
from scipy.io import savemat
from config_ENV import CONFIG_ENV
savemat("CONFIG_ENV.mat",CONFIG_ENV)
LOG_STD_MAX = 2
LOG_STD_MIN = -5
SPS=40

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env.seed(seed)
        # env.action_space.seed(seed)
        # env.observation_space.seed(seed)
        return env

    return thunk

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling

        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

if __name__ == '__main__':
    g = gclib.py()
    g.GOpen("192.168.255.200 --direct -s ALL")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = gym.vector.SyncVectorEnv([make_env("RL_/CustomEnv-v0", 1, 0, False, "policy test")])
    envs.single_observation_space.dtype = np.float32
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    actor = Actor(envs).to(device)

    path="./wandb/run-20230823_180945-dibfzq19/files"
    actor.load_state_dict(torch.load(f"{path}/actor_final.pt"))
    actor.eval()
    obs = envs.reset()

    for global_step in range(5000):
        t=wpt.time()
        obs=np.array([envs.envs[0].state]) #Read state before deciding action
        
        _, _, actions = actor.get_action(torch.Tensor(obs).to(device))
        actions = actions.detach().cpu().numpy()
        _, _, dones, infos = envs.step(actions)
        while wpt.time()-t<1/SPS:
            pass
    envs.close()
    g.GClose()