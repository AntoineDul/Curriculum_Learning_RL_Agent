import numpy as np

from source.config import CONFIG

class Block:
    def __init__(self) -> None:
        self.position: tuple[int, int] = (0, 0)
        self.reset()

    def reset(self):
        grid_size = CONFIG["grid_size"]
        self.position = tuple(int(x) for x in np.random.randint(1, (grid_size - 1), size=2, dtype=int))
        print("RESEEETTTTT")

    def set_position(self, new_position: tuple):
        self.position = new_position

    def on_edge(self):
        grid_size = CONFIG["grid_size"]
        x, y = self.position
        if x <= 0 or x >= grid_size - 1 or y <= 0 or y >= grid_size - 1 :
            print("EDGEEEEEEE")
            return True
        else:
            return False


