from dataclasses import dataclass
from typing import TypeVar, Generic

import gymnasium as gym
import numpy as np

#from parameters import rewards
from shared.rewards.components import RewardComponents
from shared.rewards.rewards import Rewards
from simulator.bittle_sim import BittleParameters, BittleSimulator



@dataclass
class EnvironmentParameters:
    length_joint_history: int = 20

    joint_min: int = -120
    joint_max: int = 120

    episode_length: int = 250

T = TypeVar("T", bound=EnvironmentParameters)


class BaseBittleEnvironment(gym.Env, Generic[T]):

    def __init__(self, parameters: T = EnvironmentParameters(), rewards: Rewards = Rewards()):
        super().__init__()

        self.params = parameters
        self.rewards = rewards

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
        #print('action:', action)
        decoded_action = self.decode_action(action)
        # print('decoded:', decoded_action)

        self.last_position = self.sim.get_position()
        self.initial_rotation = self.sim.get_rotation_matrix()

        self.sim.step(decoded_action)

        joint_angles = self.sim.get_joint_angles()
        self.update_joint_history(joint_angles)
        
        
        observation = self.get_observation()

        components = self.get_reward_components()
        reward, penalty = self.get_reward(components=components)

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
        # print('moving:', self.sim.is_moving())

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

    def get_reward_components(self) -> RewardComponents:
        position = self.sim.get_position()
        velocity = self.sim.world_to_local(self.sim.get_velocity())

        jitter_1st_order, jitter_2nd_order = self.get_joint_jitter()

        imu_gyro = self.sim.get_sensor(self.sim.params.sensor_gyro)
        imu_accel = self.sim.get_sensor(self.sim.params.sensor_accel)

        paw_clearance = self.sim.get_feet_z()
        num_arms_contacting = self.sim.get_num_contacting_arms()

        roll, pitch = self.sim.get_tilt()

        return RewardComponents(
            position=position,
            velocity=velocity,

            jitter_1st_order=jitter_1st_order,
            jitter_2nd_order=jitter_2nd_order,

            imu_gyro=imu_gyro,
            imu_accel=imu_accel,

            paw_clearance=paw_clearance,
            num_arms_contacting=num_arms_contacting,

            roll=roll,
            pitch=pitch
        )
    
    def get_reward(self, components: RewardComponents):
        # Calculate movement reward and penalty
        position_delta = components.position - self.last_position
        local_position_delta = self.initial_rotation @ position_delta
        

        movement_reward = self.rewards.WT_Movement * local_position_delta[0]
        movement_penalty = self.rewards.WT_LateralMovement * abs(local_position_delta[1])

        #velocity_reward = self.rewards.WT_Velocity * components.velocity[0]
        velocity_penalty = self.rewards.WT_Velocity * abs(components.velocity[1])

        progress_penalty = 0
        if local_position_delta[0] < 0.01:
            progress_penalty = 1
        

        # Calculate joint jitter penalty
        smooth_movement_penalty = self.rewards.WT_Smooth * np.sum(components.jitter_1st_order**2 + components.jitter_2nd_order**2)

        # Calculate clearance penalty
        clearance_penalty = self.rewards.WT_Clearance * sum( [max(0, foot_z - self.rewards.PAW_Z_THRESHOLD) for foot_z in components.paw_clearance.values()] )

        # Calculate crawling penalty
        crawling_penalty = self.rewards.WT_Crawl * components.num_arms_contacting

        # Calculate tilt penalty        
        stability_angle_penalty = self.rewards.WT_Instability * (components.roll**2 + components.pitch**2)
        stability_rate_penalty = self.rewards.WT_Instability * (components.imu_gyro[0]**2 + components.imu_gyro[1]**2)
        z_bounce_penalty = self.rewards.WT_Instability * self.rewards.WT_Bouncing * np.abs(components.velocity[2])
        
        # Calculate inactivity penalty
        inactive_penalty = 0
        if not self.sim.is_moving():
            inactive_penalty = self.rewards.WT_Inactive
        

        reward = {
            'movement': movement_reward,
            #'velocity': velocity_reward,
        }

        penalty = {
            'movement': movement_penalty,
            'velocity': velocity_penalty,
            'smooth': smooth_movement_penalty,
            'clearance': clearance_penalty,
            'crawling': crawling_penalty,
            'stability_angle': stability_angle_penalty,
            'stability_rate': stability_rate_penalty,
            'z_bounce': z_bounce_penalty,
            'living': self.rewards.WT_Living,
            'inactive': inactive_penalty,
            'progress': progress_penalty
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
    