from RL_.envs.CustomEnv import CustomEnv
from custom_callbacks import CustomCallbacks
from replay_buffer import CustomReplayBuffer

CONFIG_SAC={

        "log_level": "INFO",
        "env": CustomEnv,
        "num_gpus": 0,
        "num_workers": 0, 
        "explore": True,
        "exploration_config": {
            "type": "StochasticSampling",
        },
    "framework": "torch",
    "replay_buffer_config": {"type" :CustomReplayBuffer,},
    # "callbacks":CustomCallbacks,
    "prioritized_replay": True,
    "gamma": 1,
}

CONFIG_TRAIN={
    "N_epoch":200
}

