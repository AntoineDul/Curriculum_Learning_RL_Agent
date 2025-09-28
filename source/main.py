from environment import GridWorld
from test_agent import RandomAgent

def main():
    env = GridWorld(n=10, max_steps=100)
    agent = RandomAgent()
   
    obs = env._get_obs()
    done = False
    iterations = 0

    while not done:
        action = agent.get_action()
        obs, reward, done = env.step(action)
        env.render()
        iterations += 1
        print(f"Obs : {obs}, Reward : {reward}, Action: {action}, Iter: {iterations}")


if __name__ == "__main__":
    main()



