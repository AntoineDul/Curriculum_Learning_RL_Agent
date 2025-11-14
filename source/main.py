from source.modules.environment.grid_world import GridWorld
from source.modules.agent.random_agent import RandomAgent

def main():
    env = GridWorld()
    agent = RandomAgent()
   
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



