from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from environment.base_bittle_environment import BaseBittleEnvironment, EnvironmentParameters


@dataclass
class ContinuousEnvironmentParameters(EnvironmentParameters):
    joint_max_delta: int = np.deg2rad(30)


class ContinuousBittleEnvironment(BaseBittleEnvironment[ContinuousEnvironmentParameters]):

    

    def __init__(self, parameters: ContinuousEnvironmentParameters = ContinuousEnvironmentParameters(), weights = {}):
        super().__init__(parameters=parameters, weights=weights)

        self.action_space = gym.spaces.Box(
            # np.array([self.params.joint_min] * self.sim.NUM_JOINTS),
            # np.array([self.params.joint_max] * self.sim.NUM_JOINTS)
            np.array([-1] * self.sim.NUM_JOINTS),
            np.array([1] * self.sim.NUM_JOINTS)
        )

        self.observation_space = gym.spaces.Dict({
            # 'joint_history': gym.spaces.Box(-1, 1, shape=(self.params.length_joint_history, self.sim.NUM_JOINTS), dtype=np.float32),
            'joint_history': gym.spaces.Box(-1, 1, shape=(self.sim.params.length_joint_history, self.sim.NUM_JOINTS), dtype=np.float32),
            'gyro': gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
            'accel': gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        })


    def decode_action(self, action):
        decoded = np.deg2rad(action * self.sim.params.joint_max)

        return decoded
