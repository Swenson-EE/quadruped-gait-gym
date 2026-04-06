from simulator.modules.reward.reward_subsystem import RewardSubsystem
from simulator.core.registry import register_module
from simulator.modules.reward_module import Reward

from simulator.modules.physics_module import Physics
from simulator.modules.physics.kinematics import Kinematics
from simulator.modules.physics.kinematics_systems.basis_kinematics import BasisKinematics
from simulator.modules.physics.sensors import Sensors

import numpy as np


@register_module(Reward)
class StabilityReward(RewardSubsystem):

    def initialize(self):
        self._weight['penalty'] = {
            'stability_angle': 1.0,
            'stability_rate': 1.0
        }

        self._normalization_factor['penalty'] = {
            'stability_rate': 0.1
        }

        self._reducers['penalty'] = {
            'stability_angle': np.sum,
            'stability_rate': np.sum
        }

    def _get_components(self):
        
        physics = self.sim.get(Physics)

        kinematics = physics.get(Kinematics)
        kn_basis = kinematics.get(BasisKinematics)

        roll, pitch = kn_basis.get_tilt()
        
        imu_gyro = physics.get(Sensors).imu_gyro.internal.get()


        reward = None
        penalty = {
            # "stability_angle": roll + pitch,
            # "stability_rate": imu_gyro[0] + imu_gyro[1]
            "stability_angle": np.abs([roll, pitch]),
            "stability_rate": np.abs([imu_gyro[0], imu_gyro[1]])
        }

        return reward, penalty
