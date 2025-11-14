import random
import keyboard

import numpy as np
from source.config import CONFIG
from source.modules.environment.block import Block
from source.modules.environment.goal import Goal
from source.modules.environment.random_reward_object import RandomRewardObject

class GridWorld:
    def __init__(self):
        self.mode = CONFIG["mode"]
        self.debug = CONFIG["debug_mode_enabled"]
        self.n = CONFIG["grid_size"]
        self.max_steps = CONFIG["max_steps"]
        self.lbda = CONFIG["lambda"]
        self.nb_obstacles = CONFIG["nb_obstacles"]
        self.goal = Goal(n=self.n, is_moving=CONFIG["goal_is_moving"], move_frequency=CONFIG["goal_move_frequency"], debug=CONFIG["debug_mode_enabled"])
        self.done = False

        # Initialize block if in block mode
        self.block = None
        self.pushing = False
        if self.mode == "block":
            self.block = Block()

        # Initialize random reward cell if in random reward object mode
        self.r_reward = None
        if self.mode == "random_reward_object":
            self.r_reward = RandomRewardObject()

        # Set up environment
        self.reset()
 
    def reset(self):
        """
        Reset environment:
            - Gives a random position to the agent and goal
            - Add obstacles in random places of the grid        
        """

        # Reset agent, goal positions
        self.agent_pos = tuple(int(x) for x in np.random.randint(0, self.n, size=2, dtype=int))
        self.goal.reset()

        # Reset block 
        if self.mode == "block":
            self.block.reset()

        # Reset Random Reward Object
        if self.mode == "random_reward_object":
            self.r_reward.reset()

        # Add random obstacles 
        self.obstacles = set()
        for _ in range(self.nb_obstacles):
            pos = tuple(int(x) for x in np.random.randint(0, self.n, size=2, dtype=int))
            if pos != self.agent_pos and pos != self.goal.position:
                self.obstacles.add(pos) 

        # Initialize the number of steps 
        self.steps = 0

        print(f"agent : {self.agent_pos} \ngoal : {self.goal.position}\n obstacles: {self.obstacles}")

    def _get_obs(self):  #TODO: change for diff of vectors (goal - agent pos)
        """
        Gives the current state of the environment

        Reach mode returns : difference vector from agent to goal and obstacles

        Block mode returns : difference vector from agent to goal, agent to block and block to goal

        Returns: List
        """
        gx, gy = self.goal.position
        ax, ay = self.agent_pos
        agent_to_goal_vector = (gx - ax, gy - ay)

        if self.mode == "block":
            bx, by = self.block.position
            agent_to_block_vector = (bx - ax, by - ay)
            block_to_goal_vector = (gx - bx, gy - by)
            return [agent_to_goal_vector, agent_to_block_vector, block_to_goal_vector, self.obstacles]

        elif self.mode == "random_reward_object":
            if self.r_reward.over:
                agent_to_random_reward_vector = None
            else:
                rx, ry = self.r_reward.position
                agent_to_random_reward_vector = (rx - ax, ry - ay)
            return [agent_to_goal_vector, agent_to_random_reward_vector, self.obstacles]

        else:
            return [agent_to_goal_vector, self.obstacles]

    def step(self, agent_action):
        """
        Executes the action chosen by the agent and updates the environment
        So far, agent moves before goal

        Inputs:
            agent_action (int) : action to execute (0 = up, 1 = right, 2 = down, 3 = left)

        Returns:
            tuple[observations, reward]:
                first element is the observation of the environment after the action,
                the second is the reward earned by the agent from the action
        """

        # Check if action is successful
        if random.random() > self.lbda:
            agent_action = np.random.choice([0, 1, 2, 3])

            # Debug
            if self.debug:
                print("Environment chooses a random action")

        # Check if action is legal and handle it
        self.pushing = False
        legal, new_agent_position = self.is_legal(self.agent_pos, agent_action)

        if legal:

            # Update block
            if self.mode == "block" and self.pushing:
                dx, dy = (new_agent_position[0] - self.agent_pos[0], new_agent_position[1] - self.agent_pos[1])
                bx, by = self.block.position
                new_block_position = (bx + dx, by + dy)
                self.block.set_position(new_block_position)
            
            # Update agent position
            self.agent_pos = new_agent_position

            # Update steps
            self.steps += 1
            reward = 0

            # Check max steps
            if self.steps >= self.max_steps:
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

            # Check if agent is on random reward cell
            if self.mode == "random_reward_object":
                if self.agent_pos == self.r_reward.position:
                    reward = self.r_reward.get_random_reward()
                    if CONFIG["max_nb_random_rewards"] == -1 or self.r_reward.count < CONFIG["max_nb_random_rewards"]:
                        self.r_reward.reset()
                    else:
                        self.r_reward.set_position(None, None)

            # Check if episode is over according to mode
            # Block mode
            if self.mode == "block":
                if self.block.position == self.goal.position:
                    reward = 1
                    self.done = True
                elif self.block.on_edge():
                    print("Block position:",self.block.position)
                    print(self.block.on_edge() == True)
                    self.block.reset()
            
            # Reach mode and random reward object mode
            else:
                gx, gy = self.goal.position
                ax, ay = self.agent_pos
                if (ax - gx == 0) and (ay - gy == 0):
                    reward = 1
                    self.done = True

        # Action illegal
        else:
            if self.debug:
                print("skip turn")
            reward = 0

        return self._get_obs(), reward

    def is_legal(self, initial_position, action, step_size=1, recursive_check=True):
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
        if action == 0:  # Left
            new_position = tuple([initial_position[0], initial_position[1] - step_size])
        elif action == 1:  # Down
            new_position = tuple([initial_position[0] + step_size, initial_position[1]])
        elif action == 2:  # Right
            new_position = tuple([initial_position[0], initial_position[1] + step_size])
        elif action == 3:  # Up
            new_position = tuple([initial_position[0] - step_size, initial_position[1]])
        else:
            return (False, None)  # Invalid action

        # Determine if agent is trying to push block
        if self.mode == "block":
            if new_position == self.block.position:
                self.pushing = True

        print(f"PUSING?: {self.pushing}\nRecursive checks: {recursive_check}")
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

        # Block check 
        if self.mode == "block" and self.pushing and recursive_check:
            print(f"--- checking if block ({self.block.position}) can do action {action} --- \nResult of function {self.is_legal(self.block.position, action, 1, False)}")
            new_block_pos_legal, _ = self.is_legal(self.block.position, action, step_size, False)
            if not new_block_pos_legal:
                if self.debug:
                    print(new_position)
                    print("Cannot push block this direction!")
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

        if self.mode == "block":
            bx, by = self.block.position
            grid[bx, by] = "B"  # Block

        if self.mode == "random_reward_object":
            rx, ry = self.r_reward.position
            grid[rx, ry] = "R"  # Random Reward Object 

        for obstacle in self.obstacles:  # Obstacles
            ox, oy = obstacle
            grid[ox, oy] = "X"

        # Print grid
        for row in grid:
            print(" ".join(row))
        
        if self.debug:
            print(f"Agent position: {self.agent_pos}\nGoal Position: {self.goal.position}\nObstacles: {self.obstacles}\nBlock position: {self.block.position}")

    # Debug helper function to get directions
    def get_user_action():
        print("Choose a direction: (w=up, s=down, a=left, d=right)")
        while True:
            key = input("> ").strip().lower()
            if key == "a":
                return 0  # LEFT
            elif key == "d":
                return 2  # RIGHT
            elif key == "w":
                return 3  # UP
            elif key == "s":
                return 1  # DOWN
            else:
                print("Invalid input. Use w/a/s/d.")