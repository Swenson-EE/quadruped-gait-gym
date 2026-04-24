from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from environment.base_bittle_environment import BaseBittleEnvironment, EnvironmentParameters
from shared.rewards.rewards import RewardWeights




class DiscreteBittleEnvironment(BaseBittleEnvironment):

    def __init__(self, parameters: EnvironmentParameters = EnvironmentParameters(), weights = RewardWeights()):
        super().__init__(parameters=parameters, weights=weights)
        
        self.num_joint_bins = 7

        self.joint_bins = np.linspace(-1, 1, self.num_joint_bins)
        
        self.action_space = gym.spaces.MultiDiscrete([len(self.joint_bins)] * self.sim.NUM_JOINTS)



    def decode_action(self, action):
        joint_targets = self.joint_bins[action]
        
        return joint_targets

