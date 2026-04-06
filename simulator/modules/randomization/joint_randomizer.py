from simulator.core.subsystem.subsystem import Subsystem
from simulator.core.registry import register_module
from simulator.modules.randomization_module import Randomization

from simulator.modules.physics_module import Physics
from simulator.modules.physics.kinematics import Kinematics
from simulator.modules.physics.kinematics_systems import JointKinematics


import numpy as np


@register_module(Randomization, noise=(-15, 15))
class JointRandomizer(Subsystem):
    #_initial_position = [1, 0, 1, 0, 1, 0, 1, 0]
    _initial_position = [0.5, 0] * 4

    def __init__(self, sim, noise):
        super().__init__(sim)
        self.joint_qpos_ids = None
        self.noise = noise

    def initialize(self):
        self.joint_qpos_ids = self.sim.robot_info.joint_qpos_ids

    def reset_start(self, rng):
        
        
        random_joint_angles_deg = rng.uniform(
            *self.noise,
            size=len(self.joint_qpos_ids)
        )
        
        kn_joint = self.sim.get(Physics).get(Kinematics).get(JointKinematics)
        kn_joint.set_angles(self._initial_position + np.deg2rad(random_joint_angles_deg))
        #kn_joint.set_angles(np.deg2rad(random_joint_angles_deg))


