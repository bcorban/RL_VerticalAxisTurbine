from RL_.envs.CustomEnv import CustomEnv
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

env = CustomEnv()
check_env(env)