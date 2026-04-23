from simulator.modules.reward.reward_subsystem import RewardSubsystem
from simulator.core.registry import register_module
from simulator.modules.reward_module import Reward

import numpy as np


@register_module(Reward)
class ActionReward(RewardSubsystem):
    """
    Penalize action change to discourage rapid motion
    """
    
    def initialize(self):
        self._reducers["penalty"]["action_smooth"] = np.sum

    def reset_end(self, rng):
        self.prev_action = None
        self.next_action = None

    def step_start(self, rng, action):
        self.prev_action = self.next_action

    def step_end(self, rng, action):
        self.curr_action = action
        self.next_action = action


    def _get_components(self):
        if self.prev_action is None:
            action_delta = 0.0
        else:
            action_delta = self.curr_action - self.prev_action

        reward = None
        penalty = {
            "action_smooth": action_delta**2
        }

        return reward, penalty

