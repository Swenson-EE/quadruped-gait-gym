from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from environment.base_bittle_environment import BaseBittleEnvironment, EnvironmentParameters


@dataclass 
class DiscreteEnvironmentParameters(EnvironmentParameters):
    joint_bin_size = 15
    num_joint_bins = 7

class DiscreteBittleEnvironment(BaseBittleEnvironment[DiscreteEnvironmentParameters]):

    def __init__(self, parameters: DiscreteEnvironmentParameters = DiscreteEnvironmentParameters(), weights = {}):
        super().__init__(parameters=parameters, weights=weights)
        
        self.joint_bins = np.linspace(-1, 1, self.params.num_joint_bins)

        
        self.action_space = gym.spaces.MultiDiscrete([len(self.joint_bins)] * self.sim.NUM_JOINTS)



    def decode_action(self, action):
        joint_targets = self.joint_bins[action]
        
        return joint_targets

