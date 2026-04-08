from simulator.core.transforms import AngleTransformableBuffer

from simulator.core.registry import SubsystemRegistry
from simulator.core.subsystem import ModularSubsystem

from simulator.modules.physics_module import Physics
from simulator.modules.physics import Kinematics
from simulator.modules.physics.kinematics_systems import JointKinematics


@SubsystemRegistry.register
class SimulationState(ModularSubsystem):

    joints = None

    def initialize(self):
        length_history = self.sim.params.length_joint_history
        
        self.joints = AngleTransformableBuffer(
            size=(length_history, 8),
            scale=100.0
        )

    def reset_start(self, rng):
        self.joints.clear()

    def step_end(self, rng):
        kinematics = self.sim.get(Physics).get(Kinematics)
        kn_joint = kinematics.get(JointKinematics)

        joint_angles = kn_joint.get_angles()
        self.joints.real.deg.push(joint_angles)