import mujoco

from simulator.core import SimulationState
from simulator.physics.state.physics_context import PhysicsContext

class PhysicsState(SimulationState):
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.context = PhysicsContext(self.model, self.data)

    
        


