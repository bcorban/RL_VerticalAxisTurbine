# from gymnasium.envs.registration import register
from gym.envs.registration import register
# register(
#     id="RL_/CustomEnv-v0",
#     entry_point="RL_.envs:CustomEnv",
# )

register(
    id="RL_/CustomEnv-v0-phase",
    entry_point="RL_.envs:CustomEnv_phase",
)