import math
import random
import numpy as np
# import gym

from source.modules.environment.goal import Goal

class GridWorld:
    def __init__(self, n=5, max_steps=100, lbda=0.9, obstacle_density=0.2, debug=False):
        self.n = n
        self.max_steps = max_steps
        self.lbda = lbda
        self.obstacle_density = obstacle_density  
        self.goal = Goal(n= self.n, is_moving=False, move_frequency=5, debug=debug)
        self.done = False
        self.debug = debug

    def reset(self):
        """
        Reset environment:
            - Gives a random position to the agent and goal
            - Add obstacles in random places of the grid        
        """
        # Define agent position
        self.agent_pos = tuple(int(x) for x in np.random.randint(0, self.n, size=2, dtype=int))
        self.goal.reset()
        self.done = False

        # Add random obstacles 
        self.obstacles = set()
        for _ in range(math.floor(self.obstacle_density * (self.n**2) )):
            pos = tuple(int(x) for x in np.random.randint(0, self.n, size=2, dtype=int))
            if pos != self.agent_pos and pos != self.goal.position:
                self.obstacles.add(pos) 

        # Initialize the number of steps 
        self.steps = 0

        if self.debug:
            print(f"agent : {self.agent_pos} \ngoal : {self.goal.position}\n obstacles: {self.obstacles}")
        
        return self._get_obs(), None

    def _get_obs(self):
        """
        Observation = [dot(goal-agent, up), dot(goal-agent, right),
                    dot(goal-agent, down), dot(goal-agent, left),
                    distance/n]
        """
        gx, gy = self.goal.position
        ax, ay = self.agent_pos

        diff = np.array([gx - ax, gy - ay], dtype=np.float32)
        norm = np.linalg.norm(diff)

        if norm != 0:
            diff_unit = diff / norm
        else:
            diff_unit = np.zeros(2, dtype=np.float32)

        # Final obs
        obs = np.concatenate([diff_unit, [norm / 3]]).astype(np.float32)
    
        return obs

    def step(self, agent_action):
        """
        Executes the action chosen by the agent and updates the environment
        So far, agent moves before goal

        Inputs:
            agent_action (int) : action to execute (0 = up, 1 = right, 2 = down, 3 = left)

        Returns:
            tuple[obs, reward]:
                first element is the observation of the environment after the action,
                the second is the reward earned by the agent from the action
        """

        # Check if action is successful
        if random.random() > self.lbda:
            agent_action = np.random.choice([0, 1, 2, 3])

            # Debug
            if self.debug:
                print("Environment chooses a random action")
        
        reward = 0

        # Check action legal and handles it
        legal, new_agent_position = self.is_legal(self.agent_pos, agent_action)
        if legal:
            self.agent_pos = new_agent_position

            # Update steps
            self.steps += 1

            # Check goal
            if self.agent_pos == self.goal.position:
                reward = 1
                self.done = True

            # Move goal if needed
            if self.goal.is_moving and self.steps % self.goal.move_frequency == 0:
                
                # Asks goal where it wants to move 
                goal_action = np.random.choice([0, 1, 2, 3])
                
                # Compute action and checks if it is legal
                legal, new_goal_position = self.is_legal(self.goal.position, goal_action, self.goal.step_size)

                # Update goal position if legal else do not move goal (could change -> loop or change goal.step_size)
                if legal:
                    self.goal.set_position(new_goal_position)

        # Action illegal
        else:
            if self.debug:
                print("skip turn")
            reward = 0

        return self._get_obs(), reward, self.done, None, None

    def is_legal(self, initial_position, action, step_size=1):
        """
        Determine if an action is legal and returns resulting position 

        Args: 
            initial_position (np.array) : starting position 
            action (int) : action to execute
            step_size (int) : size of the displacement
            
        Returns:
            tuple[bool, np.array]:
                first element indicates if action is legal, second returns new position
        """
        new_position = None
        if action == 0:  # Up
            new_position = tuple([initial_position[0], initial_position[1] - step_size])
        elif action == 1:  # Right
            new_position = tuple([initial_position[0] + step_size, initial_position[1]])
        elif action == 2:  # Down
            new_position = tuple([initial_position[0], initial_position[1] + step_size])
        elif action == 3:  # Left
            new_position = tuple([initial_position[0] - step_size, initial_position[1]])
        else:
            return (False, None)  # Invalid action

        # Boundary check
        if not (0 <= new_position[0] < self.n and 0 <= new_position[1] < self.n):
            # Debug
            if self.debug: 
                print(new_position)
                print("Illegal move, boundary!")
            return (False, None)

        # Obstacle check
        if new_position in self.obstacles:
            if self.debug:
                print(new_position)
                print("Illegal move, obstacle!")
            return (False, None)

        return (True, new_position)

    def render(self):
        """
        Render the environment as a string
        """
        grid = np.full((self.n, self.n), ".")
        ax, ay = self.agent_pos
        gx, gy = self.goal.position
        grid[ax, ay] = "A"  # Agent
        grid[gx, gy] = "G"  # Goal

        for obstacle in self.obstacles:  # Obstacles
            ox, oy = obstacle
            grid[ox, oy] = "X"

        # Print grid
        print()
        for row in grid:
            print(" ".join(row))
