from simulator.physics.core import RandomizationSubsystem
import numpy as np

class JointRandomizer(RandomizationSubsystem):
    def __init__(self, sim, joint_qpos_ids, noise=(-15, 15)):
        super().__init__(sim)
        self.joint_qpos_ids = joint_qpos_ids
        self.noise = noise

    def apply(self, rng):
        random_joint_angles_deg = rng.uniform(
            *self.noise,
            size=len(self.joint_qpos_ids)
        )
        
        self.context.systems.kinematics.joint.set_angles(np.deg2rad(random_joint_angles_deg))


