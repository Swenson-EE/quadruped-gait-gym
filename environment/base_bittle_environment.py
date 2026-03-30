from dataclasses import asdict, dataclass
from typing import TypeVar, Generic

import gymnasium as gym
import numpy as np

#from parameters import rewards
from shared.rewards.components import RewardComponents, RewardNormalizationFactors, normalize_reward_components
from shared.rewards.rewards import Rewards
from simulator.bittle_sim import BittleParameters, BittleSimulator

from shared.pickle.pickle_fields import find_unpickable_fields


@dataclass
class EnvironmentParameters:
    length_joint_history: int = 20

    joint_min: int = -120
    joint_max: int = 120

    episode_length: int = 250
    total_length: int = 1e6

    normalization_factors: RewardNormalizationFactors = RewardNormalizationFactors(
        position=10.0,
        velocity=10.0,

        jitter_1st_order=1.0,
        jitter_2nd_order=0.5,

        imu_gyro=100.0,
        imu_accel=100.0,

        paw_clearance=1.0,
        paw_slipping=1.0,
        num_arms_contacting=1.0,

        roll=10.0,
        pitch=10.0
    )

    rewards: Rewards = None


T = TypeVar("T", bound=EnvironmentParameters)


class BaseBittleEnvironment(gym.Env, Generic[T]):

    def __init__(self, parameters: T = EnvironmentParameters()):
        super().__init__()

        self.params = parameters

        if self.params.rewards is None:
            self.params.rewards = Rewards.from_json_file('config/rewards.json')
        

        bittle_params = BittleParameters(
            model_path="model/bittle_mujoco.xml"
        )

        self.sim = BittleSimulator(parameters=bittle_params)
        self.OBSERVATION_DIM = (self.params.length_joint_history * self.sim.NUM_JOINTS) + self.sim.NUM_IMU_OBS

        self.step_count = 0
        self.total_session_step_count = 0

        self.total_distance_traveled = 0

        self.history_id = 0
        self.joint_history = np.zeros((self.params.length_joint_history, self.sim.NUM_JOINTS))



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.joint_history = np.zeros((self.params.length_joint_history, self.sim.NUM_JOINTS))
        self.history_id = 0

        self.step_count = 0
        self.total_distance_traveled = 0
        self.last_position = np.zeros(3)

        self.sim.reset()


        

        self.sim.randomization.apply(self.np_random)
        self.sim.forward()

        self.sim.data.qpos[2] -= np.min(self.sim.context.kinematics.foot.paw_clearance())
        self.sim.data.qvel = 0

        self.sim.forward()
             

        

        observation = self.get_observation()
        info = {}

        #print('Observation (reset):', observation)
        
        return observation, info


        

    def step(self, action):
        #print('action:', action)
        decoded_action = self.decode_action(action)
        #print('decoded:', decoded_action)

        self.last_position = self.sim.context.kinematics.world.get_position().copy()

        self.sim.step(decoded_action)
        
        joint_angles = self.sim.context.kinematics.joint.get_angles()
        self.update_joint_history(joint_angles)
        
        
        observation = self.get_observation()

        components = self.get_reward_components()
        #normalized_components = normalize_reward_components(components, self.params.normalization_factors)
        reward, penalty = self.get_reward(components=components)

        info = self.get_info()
        info['reward'] = reward
        info['penalty'] = penalty
        info['observation'] = observation
        info['components'] = asdict(components)

        total_reward = sum(reward.values()) - sum(penalty.values())

        terminated = False
        truncated = False

        self.step_count += 1
        if self.step_count > self.params.episode_length:
            self.total_session_step_count += self.step_count
            terminated = False
            truncated = True

        elif self.sim.context.metrics.is_fallen():
            self.total_session_step_count += self.step_count
            total_reward = 0
            terminated = True
            truncated = False

        # print('Observation (step):', observation)
        # print('moving:', self.sim.is_moving())

        return observation, total_reward, terminated, truncated, info
    

    @property
    def rewards(self):
        return self.params.rewards


    def get_observation(self):
        imu_gyro = self.sim.context.sensors.imu_gyro
        imu_accel = self.sim.context.sensors.imu_accel

        joint_history = self.get_joint_obs()

        return {
            "joint_history": joint_history,
            "gyro": imu_gyro,
            "accel": imu_accel
        }

    def get_reward_components(self) -> RewardComponents:
        position = self.sim.context.kinematics.world.get_position()
        velocity = self.sim.context.kinematics.basis.world_to_local(self.sim.context.kinematics.world.get_velocity())

        jitter_1st_order, jitter_2nd_order = self.get_joint_jitter()

        imu_gyro = self.sim.context.sensors.imu_gyro
        imu_accel = self.sim.context.sensors.imu_accel

        paw_clearance = self.sim.context.kinematics.foot.paw_clearance()
        paw_slipping, num_paws_contacting = self.sim.context.kinematics.foot.paw_slipping()
        num_arms_contacting = len(self.sim.context.contacts.contacting_geoms(self.sim.context.robot_info.arm_geom_ids))

        roll, pitch = self.sim.context.kinematics.basis.get_tilt()

        joint_variance = np.var(self.get_joint_obs()[-10:], axis=0)
        

        return RewardComponents(
            position=position,
            velocity=velocity,

            jitter_1st_order=jitter_1st_order,
            jitter_2nd_order=jitter_2nd_order,

            imu_gyro=imu_gyro,
            imu_accel=imu_accel,

            paw_clearance=paw_clearance,
            paw_slipping=paw_slipping,
            num_arms_contacting=num_arms_contacting,
            num_paws_contacting=num_paws_contacting,

            roll=roll,
            pitch=pitch,

            joint_variance=joint_variance
        )
    
    def get_reward(self, components: RewardComponents):
        # Calculate movement reward and penalty
        position_delta = components.position - self.last_position
        local_position_delta = self.sim.context.kinematics.basis.world_to_local(position_delta)

        self.total_distance_traveled += local_position_delta[0]

        movement_reward = self.rewards.WT_Movement * local_position_delta[0]
        movement_penalty = self.rewards.WT_LateralMovement * (abs(local_position_delta[1]**2) + abs(local_position_delta[2]**2))

        total_movement_reward = self.rewards.WT_Movement * self.total_distance_traveled

        #velocity_reward = self.rewards.WT_Velocity * components.velocity[0]
        velocity_penalty = self.rewards.WT_Velocity * abs(components.velocity[1])       

        # Calculate joint jitter penalty
        smooth_movement_penalty = self.rewards.WT_Smooth * np.sum(components.jitter_1st_order**2 + components.jitter_2nd_order**2)

        # Calculate clearance penalty
        clearance_penalty = self.rewards.WT_Clearance * sum( [max(0, foot_z - self.rewards.PAW_Z_THRESHOLD) for foot_z in components.paw_clearance] )

        # Calculate slip penalty
        slip_penalty = self.rewards.WT_Slip * np.sum(components.paw_slipping**2)
        
        # Calculate crawling penalty
        crawling_penalty = self.rewards.WT_Crawl * components.num_arms_contacting

        # Paw contacting penalty
        paw_contact_penalty = 0 if components.num_paws_contacting > 1 else 1

        # Calculate tilt penalty        
        stability_angle_penalty = self.rewards.WT_Instability * (components.roll**2 + components.pitch**2)
        stability_rate_penalty = self.rewards.WT_InstabilityRate * (components.imu_gyro[0]**2 + components.imu_gyro[1]**2)
        z_bounce_penalty = self.rewards.WT_Instability * self.rewards.WT_Bouncing * np.abs(components.velocity[2])
        
        # Calculate stagnation penalty
        stagnation_penalty = 100 * sum(components.joint_variance < np.deg2rad(10))
        #print('stagnation:', stagnation_penalty)

        reward = {
            'movement': movement_reward,
            'total_movement': total_movement_reward
            #'velocity': velocity_reward,
        }

        penalty = {
            'movement': movement_penalty,
            'velocity': velocity_penalty,
            'smooth': smooth_movement_penalty,
            'clearance': clearance_penalty,
            'crawling': crawling_penalty,
            'slipping': slip_penalty,
            'stability_angle': stability_angle_penalty,
            'stability_rate': stability_rate_penalty,
            'z_bounce': z_bounce_penalty,
            'living': self.rewards.WT_Living,
            'steps': self.total_session_step_count / self.params.total_length,
            'paw_contact': paw_contact_penalty,
            'stagnation': stagnation_penalty
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
    