from simulator.modules.reward.reward_subsystem import RewardSubsystem
from simulator.core.registry import register_module
from simulator.modules.reward_module import Reward

from simulator.modules.physics_module import Physics
from simulator.modules.physics.kinematics import Kinematics
from simulator.modules.physics.kinematics_systems.foot_kinematics import FootKinematics

import numpy as np


@register_module(Reward)
class AirTimeReward(RewardSubsystem):
    """
    Penalize action change to discourage rapid motion
    """

    def initialize(self):
        self.air_time = np.zeros(4)

        self._reducers["reward"]["air_time"] = np.sum

    def reset_start(self, rng):
        self.air_time = np.zeros(4)

    def step_end(self, rng, action):
        physics = self.sim.get(Physics)

        kinematics = physics.get(Kinematics)
        kn_foot: FootKinematics = kinematics.get(FootKinematics)

        _, paw_contacts = kn_foot.paw_slipping()

        for i in range(4):
            if paw_contacts[i] == 0:
                self.air_time[i] += self.sim.params.control_dt
            else:
                self.air_time[i] = 0.0

    def _get_components(self):
        reward = {
            "air_time": self.air_time
        }
        penalty = None

        return reward, penalty

