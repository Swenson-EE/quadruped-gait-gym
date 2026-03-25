from dataclasses import dataclass
from typing import TypeVar, Generic

import gymnasium as gym
import numpy as np

from parameters import rewards
from simulator.bittle_sim import BittleParameters, BittleSimulator



@dataclass
class EnvironmentParameters:
    length_joint_history: int = 20

    joint_min: int = -120
    joint_max: int = 120

    episode_length: int = 250

T = TypeVar("T", bound=EnvironmentParameters)


class BaseBittleEnvironment(gym.Env, Generic[T]):

    def __init__(self, parameters: T = EnvironmentParameters()):
        super().__init__()

        self.params = parameters

        bittle_params = BittleParameters(
            model_path="model/bittle_mujoco.xml"
        )

        self.sim = BittleSimulator(parameters=bittle_params)
        self.OBSERVATION_DIM = (self.params.length_joint_history * self.sim.NUM_JOINTS) + self.sim.NUM_IMU_OBS

        self.step_count = 0
        self.total_session_step_count = 0


        self.history_id = 0
        self.joint_history = np.zeros((self.params.length_joint_history, self.sim.NUM_JOINTS))



    def reset(self, seed=None, options=None):
        self.joint_history = np.zeros((self.params.length_joint_history, self.sim.NUM_JOINTS))
        self.history_id = 0

        self.step_count = 0
        self.sim.reset()

        self.last_position = self.sim.get_position()

        observation = self.get_observation()
        info = {}

        #print('Observation (reset):', observation)
        
        return observation, info


        

    def step(self, action):
        decoded_action = self.decode_action(action)

        self.last_position = self.sim.get_position()
        self.initial_rotation = self.sim.get_rotation_matrix()

        self.sim.step(decoded_action)

        joint_angles = self.sim.get_joint_angles()
        self.update_joint_history(joint_angles)
        
        
        observation = self.get_observation()
        reward, penalty = self.get_reward()
        info = self.get_info()
        info['reward'] = reward
        info['penalty'] = penalty
        info['observation'] = observation

        total_reward = sum(reward.values()) - sum(penalty.values())

        terminated = False
        truncated = False

        self.step_count += 1
        if self.step_count > self.params.episode_length:
            self.total_session_step_count += self.step_count
            terminated = False
            truncated = True

        elif self.sim.is_fallen():
            self.total_session_step_count += self.step_count
            total_reward = 0
            terminated = True
            truncated = False

        # print('Observation (step):', observation)

        return observation, total_reward, terminated, truncated, info
    

    def get_observation(self):
        imu_gyro = self.sim.get_sensor(self.sim.params.sensor_gyro)
        imu_accel = self.sim.get_sensor(self.sim.params.sensor_accel)
        joint_history = self.get_joint_obs()

        return {
            "joint_history": joint_history,
            "gyro": imu_gyro,
            "accel": imu_accel
        }

    
    def get_reward(self):
        current_position = self.sim.get_position()
        velocity = self.sim.world_to_local(self.sim.get_velocity())

        # Calculate movement reward and penalty
        position_delta = current_position - self.last_position
        local_position_delta = self.initial_rotation @ position_delta

        movement_reward = rewards.WT_Movement * local_position_delta[0]
        movement_penalty = rewards.WT_Movement * abs(local_position_delta[1])

        # Calculate joint jitter penalty
        jitter_1st_order, jitter_2nd_order = self.get_joint_jitter()
        smooth_movement_penalty = rewards.WT_Smooth * np.sum(jitter_1st_order**2 + jitter_2nd_order**2)


        # Calculate clearance penalty
        paw_clearance = self.sim.get_feet_z()
        clearance_penalty = rewards.WT_Clearance * sum( [max(0, foot_z - rewards.PAW_Z_THRESHOLD) for foot_z in paw_clearance.values()] )

        # Calculate crawling penalty
        num_arms_contacting = self.sim.get_num_contacting_arms()
        crawling_penalty = rewards.WT_Crawl * num_arms_contacting

        # Calculate tilt penalty
        roll, pitch = self.sim.get_tilt()
        imu_gyro = self.sim.get_sensor(self.sim.params.sensor_gyro)
        
        stability_angle_penalty = rewards.WT_Instability * (roll**2 + pitch**2)
        stability_rate_penalty = rewards.WT_Instability * (imu_gyro[0]**2 + imu_gyro[1]**2)
        z_bounce_penalty = rewards.WT_Instability * rewards.WT_Bouncing * np.abs(velocity[2])
        

        reward = {
            'movement': movement_reward,
        }

        penalty = {
            'movement': movement_penalty,
            'smooth': smooth_movement_penalty,
            'clearance': clearance_penalty,
            'crawling': crawling_penalty,
            'stability_angle': stability_angle_penalty,
            'stability_rate': stability_rate_penalty,
            'z_bounce': z_bounce_penalty
        }

        return reward, penalty
    
    def get_info(self):
        return {}
    
    def decode_action(self, action):
        return action

    def get_joint_obs(self):
        id = self.history_id
        return np.roll(self.joint_history, -id, axis=0)

    
    def get_joint_history(self, n=0, units='rad'):
        id = (self.history_id - n - 1) % self.params.length_joint_history

        if units == 'rad':
            return np.deg2rad(self.joint_history[id])
        elif units == 'deg':
            return self.joint_history[id]
        else:
            raise Exception("Unsupported units") # unsupported units
    
    def update_joint_history(self, joint_angles):
        self.joint_history[self.history_id] = joint_angles
        self.history_id = (self.history_id + 1) % self.params.length_joint_history # Move to the next position in the history (circular buffer)

    def get_joint_jitter(self):
        joint_angle_t = self.get_joint_history(n=0, units='rad')
        joint_angle_t1 = self.get_joint_history(n=1, units='rad')
        joint_angle_t2 = self.get_joint_history(n=2, units='rad')
        
        jitter_1st_order = joint_angle_t - joint_angle_t1
        jitter_2nd_order = joint_angle_t - 2 * joint_angle_t1 + joint_angle_t2

        return np.abs(jitter_1st_order), np.abs(jitter_2nd_order)
    