from simulator.modules.reward.reward_subsystem import RewardSubsystem
from simulator.core.registry import register_module
from simulator.modules.reward_module import Reward

import numpy as np

@register_module(Reward)
class SymmetryReward(RewardSubsystem):
    def initialize(self):
        self.action = None

        self.left = [0, 2, 4, 6]
        self.right = [1, 3, 5, 7]

        self._reducers["penalty"]["symmetry"] = np.sum

    def step_end(self, rng, action):
        self.action = action

        

    def _get_components(self):
        symmetry = self.action[self.left] - self.action[self.right]

        reward = None
        penalty = {
            "symmetry": symmetry
        }

        return reward, penalty