from modules.environment.grid_world import GridWorld
from modules.agent.random_agent import RandomAgent

def main():
    env = GridWorld(n=10, max_steps=100, lbda=0.99, obstacle_density=0.2, debug=True)
    agent = RandomAgent()
   
    obs = env._get_obs()
    iterations = 0

    while not env.done:
        action = agent.get_action()
        obs, reward = env.step(action)
        env.render()
        iterations += 1
        print(f"Obs : {obs}, Reward : {reward}, Action: {action}, Steps: {env.steps}, Total Iter: {iterations}")


if __name__ == "__main__":
    main()



