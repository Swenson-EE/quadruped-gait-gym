import mujoco

from simulator.state.sim_context import SimulationContext
from simulator.controllers import RandomizationController
from simulator.randomization import InitialPoseRandomizer, JointRandomizer

class RobotState:
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


