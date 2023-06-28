CONFIG_ENV={
        "N_max":100, #Number of rotations for one episode
        "N_transient_effects":1, #start applying policy but wait N rotations before sampling transitions
        "action_lb":-3, #action space bounds
        "action_hb":3,
        "bc":'001', #file parameters
        "date":'20230628'
        }