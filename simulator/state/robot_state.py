import mujoco

from simulator.state.sim_context import SimulationContext
from simulator.controllers import RandomizationController


class RobotState:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.context = SimulationContext(self.model, self.data)

        self.randomization = RandomizationController(modules=[
            
        ])


