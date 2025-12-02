import numpy as np

from source.config import CONFIG

class Block:
    def __init__(self, n) -> None:
        self.position: tuple[int, int] = (0, 0)
        self.n = n
        self.reset()

    def reset(self):
        self.position = tuple(int(x) for x in np.random.randint(1, (self.n - 1), size=2, dtype=int))

    def set_position(self, new_position: tuple):
        self.position = new_position

    def on_edge(self):
        x, y = self.position
        if x <= 0 or x >= self.n - 1 or y <= 0 or y >= self.n - 1 :
            return True
        else:
            return False


