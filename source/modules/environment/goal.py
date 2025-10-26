from source.modules.environment.environment_element import EnvironmentElement

class Goal(EnvironmentElement):

    def __init__(self, n=5, is_moving=False, move_frequency=5, step_size=1, debug=False) -> None:
        super().__init__()

    def set_position(self, new_position):
        """
        Assign a new position to the goal 
        """
        super().set_position(new_position)

        if self.debug:
            print(f"Goal position changed to {new_position}")

        return

