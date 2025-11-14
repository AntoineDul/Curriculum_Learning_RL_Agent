from source.modules.environment.environment_element import EnvironmentElement
import numpy as np
from source.config import CONFIG

class RandomRewardObject(EnvironmentElement) : 
    def __init__(self):
        super().__init__()
        self.count = 0
        self.over = False
        self.reward_range = CONFIG["reward_range"]
        low, high = self.reward_range
        self.rd_mean = (low + high) / 2
        self.rd_std = (high - low) / 4

    def get_random_reward(self):
        low, high = self.reward_range
        r = np.random.normal(self.rd_mean, self.rd_std)

        # Ensure we stay in correct range
        r = max(low, min(high, r))

    def set_random_reward_range(self, low, high):
        self.reward_range = [low, high]

    def set_rd_std(self, new_std):
        self.rd_std = new_std
    
    def set_rd_mean(self, new_mean):
        self.rd_std = new_mean

    def reset(self):
        if not self.over:
            super().reset()