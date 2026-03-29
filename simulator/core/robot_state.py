import mujoco

from simulator.core.sim_context import SimulationContext


class RobotState:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.context = SimulationContext(self.model, self.data)

