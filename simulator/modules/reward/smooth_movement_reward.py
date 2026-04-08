from simulator.modules.reward.reward_subsystem import RewardSubsystem
from simulator.core.registry import register_module
from simulator.modules.reward_module import Reward

from simulator.modules.simulation_state import SimulationState

import numpy as np


@register_module(Reward)
class SmoothMovementReward(RewardSubsystem):

    def initialize(self):
        self._normalization_factor['penalty'] = {
            'jitter_1st': 0.1,
            'jitter_2nd': 0.1
        }

        self._reducers['penalty']['jitter_1st'] = np.sum
        self._reducers['penalty']['jitter_2nd'] = np.sum

    def _get_components(self):
        joints = self.sim.get(SimulationState).joints
        jitter_1st_order, jitter_2nd_order = joints.get_jitter()

        reward = None
        penalty = {
            "jitter_1st": jitter_1st_order,
            "jitter_2nd": jitter_2nd_order
        }

        return reward, penalty