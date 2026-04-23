from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from environment.base_bittle_environment import BaseBittleEnvironment, EnvironmentParameters


@dataclass
class ContinuousEnvironmentParameters(EnvironmentParameters):
    joint_max_delta: int = np.deg2rad(25)


class ContinuousBittleEnvironment(BaseBittleEnvironment[ContinuousEnvironmentParameters]):

    

    def __init__(self, parameters: ContinuousEnvironmentParameters = ContinuousEnvironmentParameters(), weights = {}):
        super().__init__(parameters=parameters, weights=weights)

        self.action_space = gym.spaces.Box(
            np.array([-1] * self.sim.NUM_JOINTS),
            np.array([1] * self.sim.NUM_JOINTS)
        )



    def decode_action(self, action):
        #joint_targets = action * self.params.joint_delta
       # decoded = np.deg2rad(action * self.sim.params.joint_max)

        return action
