from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from environment.base_bittle_environment import BaseBittleEnvironment, EnvironmentParameters


@dataclass
class ContinuousEnvironmentParameters(EnvironmentParameters):
    joint_max_delta: int = np.deg2rad(30)


class ContinuousBittleEnvironment(BaseBittleEnvironment[ContinuousEnvironmentParameters]):

    

    def __init__(self, parameters: ContinuousEnvironmentParameters = ContinuousEnvironmentParameters()):
        super().__init__(parameters=parameters)

        self.action_space = gym.spaces.Box(
            np.array([self.params.joint_min] * self.sim.NUM_JOINTS),
            np.array([self.params.joint_max] * self.sim.NUM_JOINTS)
        )

        self.observation_space = gym.spaces.Dict({
            'joint_history': gym.spaces.Box(self.params.joint_min, self.params.joint_max, shape=(self.params.length_joint_history, self.sim.NUM_JOINTS), dtype=np.float32),
            'gyro': gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            'accel': gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        })


    def decode_action(self, action):
        return action
        return np.deg2rad(action)
