CONFIG_ENV={
        "N_max":1000, #Number of rotations for one episode
        "N_transient_effects":1, #start applying policy but wait N rotations before sampling transitions
        "action_lb":-6, #action space bounds
        "action_hb":6,
        "bc":'002', #file parameters
        "date":'20230721',
        "ms":4,
        }