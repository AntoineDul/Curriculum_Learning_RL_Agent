import math
import random

import numpy as np

# import gym


class GridWorld:
    def __init__(self, n=5, max_steps=100, lbda=0.9, obstacle_density=0.2):
        self.n = n
        self.max_steps = max_steps
        self.lbda = lbda
        self.obstacle_density = obstacle_density  # TODO
        self.reset()

    def reset(self):
        """Reset environment"""
        # Define agent and goal positions
        self.agent_pos = np.random.randint(0, self.n, size=2)
        self.goal_pos = np.random.randint(0, self.n, size=2)
        
        # Add random obstacles 
        self.obstacles = []
        for _ in range(math.floor(self.obstacle_density * self.n**2)):
            pos = np.random.randint(0, self.n, size=2)
            if not (np.array_equal(pos, self.agent_pos) or np.array_equal(pos, self.goal_pos)):
                self.obstacles.append(pos)

        # Initialize the number of steps 
        self.steps = 0

    def _get_obs(self):
        """
        Gives the current state of the env

        Returns: array(agent_x, agent_y, goal_x, goal_y)
        """
        return [self.agent_pos, self.goal_pos, np.array(self.obstacles)]

    def step(self, action):
        """
        Apply an action chosen by the agent and updates env
        Actions: 0 = up, 1 = right, 2 = down, 3 = left
        Returns: (obs, reward, done)
        """

        # Check if action is successful
        if random.random() > self.lbda:
            print("no luck!")
            action = np.random.choice([0, 1, 2, 3])

        if self.is_legal(action):
            if action == 0:  # Up
                self.agent_pos[1] -= 1
            if action == 1:  # Right
                self.agent_pos[0] += 1
            if action == 2:  # Down
                self.agent_pos[1] += 1
            if action == 3:  # Left
                self.agent_pos[0] -= 1

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

        # Action illegal
        else:
            reward = -1
            done = False

        return self._get_obs(), reward, done

    def is_legal(self, action):
        new_pos = None
        if action == 0:  # Up
            new_pos = np.array([self.agent_pos[0], self.agent_pos[1] - 1])
        elif action == 1:  # Right
            new_pos = np.array([self.agent_pos[0] + 1, self.agent_pos[1]])
        elif action == 2:  # Down
            new_pos = np.array([self.agent_pos[0], self.agent_pos[1] + 1])
        elif action == 3:  # Left
            new_pos = np.array([self.agent_pos[0] - 1, self.agent_pos[1]])
        else:
            return False  # Invalid action

        # Boundary check
        if not (0 <= new_pos[0] < self.n and 0 <= new_pos[1] < self.n):
            return False

        # Obstacle check
        if any(np.array_equal(new_pos, obs) for obs in self.obstacles):
            return False

        return True

    def render(self):
        """
        Render the environment as a string
        """
        grid = np.full((self.n, self.n), ".")
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        grid[ay, ax] = "A"  # Agent
        grid[gy, gx] = "G"  # Goal

        for obstacle in self.obstacles:  # Obstacles
            ox, oy = obstacle
            grid[ox, oy] = "X"

        # Print grid
        for row in grid:
            print(" ".join(row))
