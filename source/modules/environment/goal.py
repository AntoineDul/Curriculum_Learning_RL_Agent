import numpy as np

class Goal:

    def __init__(self, n=5, is_moving=False, move_frequency=5, step_size=1, debug=False) -> None:
        self.n = n
        self.is_moving = is_moving
        self.move_frequency = move_frequency
        self.step_size = step_size
        self.debug = debug
        self.reset()

    def set_position(self, new_position):
        """
        Assign a new position to the goal 
        """
        self.position = new_position
        
        if self.debug:
            print(f"Goal position changed to {new_position}")

        return

    def reset(self):
        self.position = tuple(int(x) for x in np.random.randint(0, self.n, size=2, dtype=int))
