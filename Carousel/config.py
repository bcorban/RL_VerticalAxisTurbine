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
    "N_epoch":200
}


CONFIG_ENV={
        "N_max":10, #Number of rotations for one episode
        "N_ep_without_homing":300,
        "N_transient_effects":3, #start applying policy but wait N rotations before sampling transitions
        "action_lb":-1, #action space bounds
        "action_hb":1,
        "bc":'001', #file parameters
        "date":'20230506'
        }