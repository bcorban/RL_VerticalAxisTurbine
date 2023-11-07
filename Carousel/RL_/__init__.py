# from gymnasium.envs.registration import register
from gym.envs.registration import register
register(
    id="RL_/CustomEnv-v0",
    entry_point="RL_.envs:CustomEnv",
)

# register(
#     id="RL_/CustomEnv-v0-alpha_dot",
#     entry_point="RL_.envs:CustomEnv_alpha_dot",
# )