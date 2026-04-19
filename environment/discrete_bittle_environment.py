from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from environment.base_bittle_environment import BaseBittleEnvironment, EnvironmentParameters


@dataclass 
class DiscreteEnvironmentParameters(EnvironmentParameters):
    joint_bin_size = 15


class DiscreteBittleEnvironment(BaseBittleEnvironment[DiscreteEnvironmentParameters]):

    def __init__(self, parameters: DiscreteEnvironmentParameters = DiscreteEnvironmentParameters(), weights = {}):
        super().__init__(parameters=parameters, weights=weights)

        self.imu_bins = np.array([-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0 ])
        self.joint_bins = np.arange(
            -self.sim.params.joint_max, 
            self.sim.params.joint_max + self.params.joint_bin_size, 
            self.params.joint_bin_size
        )

        self.action_space = gym.spaces.Discrete([len(self.joint_bins)] ** self.sim.NUM_JOINTS)

        #self.action_space = gym.spaces.MultiDiscrete([len(self.joint_bins)] * 8)
        # self.observation_space = gym.spaces.Dict({
        #     'gyro': gym.spaces.MultiDiscrete([len(self.imu_bins)] * 3),
        #     'accel': gym.spaces.MultiDiscrete([len(self.imu_bins)] * 3),
        #     'joint_history': gym.spaces.MultiDiscrete(
        #         [len(self.joint_bins)] * (self.sim.NUM_JOINTS * self.sim.params.length_joint_history)
        #     )
        # })
        self.observation_space = gym.spaces.Dict({
            'joint_history': gym.spaces.Box(-1, 1, shape=(self.sim.params.length_joint_history, self.sim.NUM_JOINTS), dtype=np.float32),
            'gyro': gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
            'accel': gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        })

    def discretize_vector(self, values, bins):
        values = np.array(values)
        
        # Compute nearest bin index
        indices = np.abs(values[:, None] - bins).argmin(axis=1)
        
        # Map back to bin values
        discretized = bins[indices]
        
        return discretized, indices

    def reconstruct_from_indices(self, indices, bins):
        return bins[indices]


    def encode_action(self, joint_indices, num_bins):
        action = 0
        for i, idx in enumerate(joint_indices):
            action += idx * (num_bins ** i)
        return action
    
    def _decode_action(self, action):
        num_bins = len(self.joint_bins)
        indices = []
        for _ in range(self.sim.NUM_JOINTS):
            indices.append(action % num_bins)
            action //= num_bins
        return np.array(indices)

    def decode_action(self, action):
        joint_angles_deg = self._decode_action(action)
        decoded = np.deg2rad(joint_angles_deg)
        
        return decoded

