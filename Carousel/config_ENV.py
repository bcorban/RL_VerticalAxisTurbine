#Config file for the environment, other parameters need to be changed in the parser of the sac_continuous_action.py file


CONFIG_ENV={
        "N_max":1000, #Number of rotations for one episode
        "N_transient_effects":1, #start applying policy but wait N rotations before sampling transitions
        "action_lb":-6, #action space bounds
        "action_hb":6,
        "bc":'002', #file parameters
        "date":'20230824',
        "ms":13, 
        "ACTUATE":True, #put False for non actuated case, and set total timesteps accordingly in sac_continuous_action file
        "pre-fill-RB":False, #True to add daniel's closed loop data to the replay buffer 
        }