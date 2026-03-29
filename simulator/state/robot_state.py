import mujoco

from simulator.randomization.initial_pose_randomizer import InitialPoseRandomizer
from simulator.state.sim_context import SimulationContext
from simulator.controllers import RandomizationController


class RobotState:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.context = SimulationContext(self.model, self.data)

        self.randomization = RandomizationController(modules=[
            InitialPoseRandomizer(
                context=self.context,
                position_range=(-0.1, 0.1),
                rotation_range=(-0.5, 0.5)
            )
        ])


