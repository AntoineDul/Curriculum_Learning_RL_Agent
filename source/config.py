CONFIG = {
    "mode":"block",                 # "block" or "reach" or "random_reward_object"
    "grid_size":10,                  # Size of the environment
    "max_steps":100,                # Maximum number of steps from the agent
    "lambda":1,                   # Lambda parameter
    "debug_mode_enabled":True,      # Debug mode (extra prints and arrow key control)
    
    # Goal config
    "goal_is_moving":False,         # Set static or moving goal 
    "goal_move_frequency":1,        # How often goal moves

    # Obstacles config
    "nb_obstacles":3,               # Number of obstacles on the grid
    "obstacles_moving":False,       # Enable obstacles movement 
    "obstacles_step_size":None,     # Step size for obstacles

    # Random Reward Object config
    "reward_range": [-1, 1],        # Range in which the reward will be chosen at random
    "max_nb_random_rewards": -1,      # Number of times the agent can get a random reward before it disappears (-1 for infinite)
}
