from simulator.modules.physics.kinematics import Kinematics
from simulator.modules.physics.kinematics_systems.joint_kinematics import JointKinematics
from simulator.modules.physics_module import Physics
from simulator.modules.reward.reward_subsystem import RewardSubsystem
from simulator.core.registry import register_module
from simulator.modules.reward_module import Reward

import numpy as np


#@register_module(Reward)
class BentJointReward(RewardSubsystem):
    _TARGET_ANGLE = 0.3 # radians
    _SIGMA = 0.1 # Tolerance band

    def initialize(self):
        #self._reducers["reward"]["bent_joint"] = np.sum
        self._reducers["reward"]["bent_joint"] = lambda x: np.mean(
            np.exp(
                -(x**2)/(2*self._SIGMA**2)
            )
        )

        self._normalization_factor["reward"]["bent_joint"] = 1.0


    def _get_components(self):
        kn_joints: JointKinematics = self.sim.get(Physics).get(Kinematics).get(JointKinematics)
        joint_angles = kn_joints.get_angles(units='rad')

        error = joint_angles - self._TARGET_ANGLE
        reward = {
            "bent_joint": error
        }
        penalty = None

        return reward, penalty
    