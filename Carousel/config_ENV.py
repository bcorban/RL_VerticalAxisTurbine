# Config file for the environment, other parameters need to be changed in the parser of the sac_continuous_action.py file

CONFIG_ENV = {
    "N_max": 500,  # Number of rotations for one episode. The setup will home and take offset after each N_max rotations. Does not impact the experiment duration.
    "N_transient_effects": 1,  # OSEF. start applying policy but wait N rotations before sampling transitions
    "action_lb": -6,  # Action space bounds
    "action_hb": 6,
    "path": "/Users/PIVUSER/Desktop/RL_VerticalAxisTurbine/Carousel/2023_BC",
    "bc": "008",  # file parameters
    "date": "20240215",
    "ms": 11,
    "ACTUATE": True,  # To run the unactuated case, first set ACTUATE to False and set total timesteps accordingly in sac_continuous_action file.
    "pre-fill-RB": False,  # True to add daniel's closed loop data to the replay buffer
    "nb_conv_time":3,
    "states_per_conv_time":3
}
