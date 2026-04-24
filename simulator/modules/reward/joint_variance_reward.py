from simulator.modules.reward.reward_subsystem import RewardSubsystem
from simulator.core.registry import register_module
from simulator.modules.reward_module import Reward

from simulator.modules.simulation_state import SimulationState

import numpy as np


@register_module(Reward)
class JointVarianceReward(RewardSubsystem):
    _var_hist = 20
    _joint_min = 15
    _joint_max = 45

    def initialize(self):
        self._normalization_factor["penalty"] = {
            # "small_joint_variance": self._joint_min,
            "large_joint_variance": self._joint_max
        }

        self._reducers["penalty"] = {
            # "small_joint_variance": lambda x: np.sum(
            #     1 - x[(x < 1)]
            # ),
            "large_joint_variance": lambda x: np.linalg.norm(
                x[x > 1]
            )
        }


    def _get_components(self):
        joints = self.sim.get(SimulationState).joints
        joint_variance = np.var(joints.real.get()[-self._var_hist:], axis=0)

        reward = None
        penalty = {
            #"small_joint_variance": joint_variance,
            "large_joint_variance": joint_variance
        }

        return reward, penalty