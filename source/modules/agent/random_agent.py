import numpy as np

class RandomAgent:

    def __init__(self) -> None:
        pass

    def get_action(self) -> int:
        return np.random.randint(low=0, high=4, dtype=int)

