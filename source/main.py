from source.modules.environment.grid_world import GridWorld
from source.modules.agent.random_agent import RandomAgent

def main():
    env = GridWorld()
    agent = RandomAgent()
   
    # Initial run
    run_environment(env, agent)

    # Change parameters and run again
    change_parameters(env, new_mode="reach", new_n=15, new_max_steps=150, new_lambda=1, enable_offset_reward=True, new_offset_value=2, new_offset_reward=1.0)
    
    # Second run
    run_environment(env, agent)

    # Change parameters and run again
    change_parameters(env, new_mode="random_reward_object", new_n=20, new_max_steps=200, new_lambda=1, enable_offset_reward=False)

    # Third run
    run_environment(env, agent)

def change_parameters(env=None, new_mode=None, new_n=None, new_max_steps=None, new_lambda=None, new_nb_obstacles=None, enable_offset_reward=None, new_offset_value=None, new_offset_reward=None, new_goal_is_moving=None, new_goal_move_frequency=None, new_goal_step_size=None):
    if env is not None:
        if new_mode is not None:
            env.set_mode(new_mode)
        if new_n is not None:
            env.set_n(new_n)
        if new_max_steps is not None:
            env.set_max_steps(new_max_steps)
        if new_lambda is not None:
            env.set_lbda(new_lambda)
        if new_nb_obstacles is not None:
            env.set_nb_obstacles(new_nb_obstacles)
        if enable_offset_reward is not None:
            env.set_offset_reward_enabled(enable_offset_reward)
        if new_offset_value is not None:
            env.set_offset_value(new_offset_value)
        if new_offset_reward is not None:
            env.set_offset_reward(new_offset_reward)
        if new_goal_is_moving is not None:
            env.goal.is_moving = new_goal_is_moving
        if new_goal_move_frequency is not None:
            env.goal.move_frequency = new_goal_move_frequency
        if new_goal_step_size is not None:
            env.goal.step_size = new_goal_step_size
        env.reset()

def run_environment(env, agent):
    obs = env._get_obs()
    iterations = 0
    env.render()

    while not env.done:
        if env.debug == True:
            action = GridWorld.get_user_action()
        else:
            action = agent.get_action()
        
        obs, reward = env.step(action)
        env.render()
        iterations += 1
        print(f"Obs : {obs}, Reward : {reward}, Action: {action}, Steps: {env.steps}, Total Iter: {iterations}")
    

if __name__ == "__main__":
    main()



