from gym.envs.registration import register
register(
    id="CustomEnv",
    entry_point="RL_.envs:CustomEnv",
)