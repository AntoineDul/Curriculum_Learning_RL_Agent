import numpy as np
import random
# import gym

class GridWorld:
    def __init__(self, n=5, max_steps=100, lbda=0.9, obstacle_density=0.2):
        self.n = n
        self.max_steps = max_steps
        self.lbda = lbda
        self.obstacle_density = obstacle_density    #TODO
        self.reset()

    def reset(self):
        """Reset environment"""
        self.agent_pos = np.random.randint(0, self.n, size=2)
        self.goal_pos = np.random.randint(0, self.n, size=2)
        self.steps = 0

    def _get_obs(self):
        """
        Gives the current state of the env

        Returns: array(agent_x, agent_y, goal_x, goal_y)
        """
        return np.array([self.agent_pos[0], self.agent_pos[1], 
                        self.goal_pos[0], self.goal_pos[1]])

    def step(self, action):
        """
        Apply an action chosen by the agent and updates env
        Actions: 0 = up, 1 = right, 2 = down, 3 = left 
        Returns: (obs, reward, done)
        """

        # Check if action is successful
        if random.random() <= self.lbda:
            if action == 0: # Up
                self.agent_pos[1] = max(self.agent_pos[1]-1, 0)
            if action == 1: # Right
                self.agent_pos[0] = min(self.agent_pos[0]+1, self.n - 1)
            if action == 2: # Down
                self.agent_pos[1] = min(self.agent_pos[1]+1, self.n - 1)
            if action == 3: # Left
                self.agent_pos[0] = max(self.agent_pos[0]-1, 0)
        
        # Choose a random action
        else :
            print("no luck!")
            action = np.random.choice([0, 1, 2, 3])

        self.steps += 1
        reward = 0
        done = False

        # Check max steps 
        if self.steps >= self.max_steps:
            done = True

        # Check goal
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = 1
            done = True

        return self._get_obs(), reward, done

    def render(self):
        """
        Render the environment as a string
        """
        grid = np.full((self.n, self.n), '.')
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        grid[ay, ax] = 'A'  # agent
        grid[gy, gx] = 'G'  # goal
        for row in grid:
            print(" ".join(row))



