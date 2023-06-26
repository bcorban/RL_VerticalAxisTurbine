CONFIG_ENV={
        "N_max":150, #Number of rotations for one episode
        "N_transient_effects":3, #start applying policy but wait N rotations before sampling transitions
        "action_lb":-1, #action space bounds
        "action_hb":1,
        "bc":'001', #file parameters
        "date":'20230506'
        }