import mujoco

from simulator.physics.state.sim_context import SimulationContext
from simulator.physics.controllers import RandomizationController
from simulator.physics.randomization import InitialPoseRandomizer, JointRandomizer

class PhysicsState:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.context = SimulationContext(self.model, self.data)

        self.randomization = RandomizationController(modules=[
            InitialPoseRandomizer(
                context=self.context
            ),
            JointRandomizer(
                context=self.context,
                joint_qpos_ids=self.context.robot_info.joint_qpos_ids
            )
        ])


