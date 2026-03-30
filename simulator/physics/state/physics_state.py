import mujoco

from simulator.physics.state.sim_context import SimulationContext

class PhysicsState:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.context = SimulationContext(self.model, self.data)

        


