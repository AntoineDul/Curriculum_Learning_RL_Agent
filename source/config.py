CONFIG = {
    "mode":"block",                 # block or reach
    "grid_size":5,                  # Size of the environment
    "max_steps":100,                # Maximum number of steps from the agent
    "lambda":0.9,                   # Lambda parameter
    "obstacle_density":0.0,         # % of obstacles out of all grid cells
    "debug_mode_enabled":True,     # Debug mode (extra prints)
    
    # Goal config
    "goal_is_moving":False,         # Set static or moving goal 
    "goal_move_frequency":1,        # How often goal moves


}
