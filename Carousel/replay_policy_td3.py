import argparse
import os
import random
import time
from distutils.util import strtobool
from typing import Sequence
()
import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from flax.training import checkpoints
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from load_mat import load_mat
import RL_
from scipy.io import savemat
from config_ENV import CONFIG_ENV
import win_precise_time as wpt
SPS=52 #frequency of actuation when replaying
savemat("CONFIG_ENV.mat", CONFIG_ENV)


def make_env(env_id):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk





class Actor(nn.Module):
    action_dim: Sequence[int]
    action_scale: Sequence[int]
    action_bias: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)
        x = x * self.action_scale + self.action_bias
        return x


class TrainState(TrainState):
    target_params: flax.core.FrozenDict

if __name__ == "__main__":
     # TRY NOT TO MODIFY: seeding
    random.seed(1)
    np.random.seed(1)
    key = jax.random.PRNGKey(1)
    key, actor_key, qf1_key, qf2_key = jax.random.split(key, 4)

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env("RL_/CustomEnv-v0-phase")])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    envs.single_observation_space.dtype = np.float32
    CKPT_DIR = '/Users/PIVUSER/Desktop/RL_VerticalAxisTurbine/Carousel/ckpts_ms007/state_step_50000'
    obs = envs.reset()

    actor = Actor(
        action_dim=np.prod(envs.single_action_space.shape),
        action_scale=jnp.array((envs.action_space.high - envs.action_space.low) / 2.0),
        action_bias=jnp.array((envs.action_space.high + envs.action_space.low) / 2.0),
    )
    
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        target_params=actor.init(actor_key, obs),
        tx=optax.adam(learning_rate=3e-4),
    )


    actor_state = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, target=actor_state)

    actor.apply = jax.jit(actor.apply)
     
    for global_step in range(5000):
        t=wpt.time()
        obs=np.array([envs.envs[0].state]) #Read state before deciding action
        if global_step<100:
            actions=np.array([0])
        else:
             actions = actor.apply(actor_state.params, obs)
             actions = np.array(
                [
                    (
                        jax.device_get(actions)[0]
                    ).clip(envs.single_action_space.low, envs.single_action_space.high)
                ]
            )

        # TRY NOT TO MODIFY: execute the game and log data.
        # next_obs, rewards, dones, infos = envs.step(actions)
        _, _, dones, infos = envs.step([[actions[0],obs[0]]])


        # ANCIEN
       #  _, _, dones, infos = envs.step([[actions[0],obs[0]]])
        while wpt.time()-t<1/SPS:
            pass

    envs.close()