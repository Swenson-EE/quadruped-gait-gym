from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from environment.base_bittle_environment import BaseBittleEnvironment, EnvironmentParameters


@dataclass 
class DiscreteEnvironmentParameters(EnvironmentParameters):
    joint_bin_size = 15
    num_joint_bins = 10

class DiscreteBittleEnvironment(BaseBittleEnvironment[DiscreteEnvironmentParameters]):

    def __init__(self, parameters: DiscreteEnvironmentParameters = DiscreteEnvironmentParameters(), weights = {}):
        super().__init__(parameters=parameters, weights=weights)

        self.joint_bins = np.arange(
            -self.sim.params.joint_max, 
            self.sim.params.joint_max + self.params.joint_bin_size, 
            self.params.joint_bin_size
        )

        print("joint bins",self.joint_bins)

        
        self.action_space = gym.spaces.MultiDiscrete([len(self.joint_bins)] * self.sim.NUM_JOINTS)



    def decode_action(self, action):
        joint_angles_deg = self.joint_bins[action]
        decoded = np.deg2rad(joint_angles_deg)
        
        return decoded

