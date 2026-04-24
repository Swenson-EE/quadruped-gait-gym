from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from environment.base_bittle_environment import BaseBittleEnvironment, EnvironmentParameters
from shared.rewards.rewards import RewardWeights




class ContinuousBittleEnvironment(BaseBittleEnvironment):

    

    def __init__(self, parameters: EnvironmentParameters = EnvironmentParameters(), weights = RewardWeights()):
        super().__init__(parameters=parameters, weights=weights)

        self.action_space = gym.spaces.Box(
            np.array([-1] * self.sim.NUM_JOINTS),
            np.array([1] * self.sim.NUM_JOINTS)
        )



    def decode_action(self, action):
        return action
