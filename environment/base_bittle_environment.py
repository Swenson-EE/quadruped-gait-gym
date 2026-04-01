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


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.sim.states.sim.joints.clear()

        self.step_count = 0
        self.total_distance_traveled = 0
        self.last_position = np.zeros(3)

        self.sim.reset()


        self.sim.randomization.apply(self.np_random)
        self.sim.forward()
        

        self.sim.data.qpos[2] -= np.min(self.sim.phys_context.kinematics.foot.paw_clearance())
        self.sim.data.qvel = 0

        self.sim.forward()
        
        self.sim.states.phys.context.sensors.reset(self.np_random)
        

        observation = self.get_observation()
        info = {}

        #print('Observation (reset):', observation)
        
        return observation, info


        

    def step(self, action):
        decoded_action = self.decode_action(action)

        self.last_position = self.sim.phys_context.kinematics.world.get_position().copy()
        self.sim.step(decoded_action)
        
        joint_angles = self.sim.phys_context.kinematics.joint.get_angles()
        self.sim.states.sim.joints.real.deg.push(joint_angles)

        self.sim.states.phys.context.sensors.step(self.np_random)

        
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

        elif self.sim.phys_context.metrics.is_fallen():
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
        imu_gyro = self.sim.phys_context.sensors.imu_gyro
        imu_accel = self.sim.phys_context.sensors.imu_accel
        joint_history = self.sim.states.sim.joints       

        return {
            "joint_history": joint_history.internal.get(),
            "gyro": np.clip(imu_gyro.internal.get(), -1.0, 1.0),
            "accel": np.clip(imu_accel.internal.get(), -1.0, 1.0)
        }

    def get_reward_components(self) -> RewardComponents:
        position = self.sim.phys_context.kinematics.world.get_position()
        position_delta = self.sim.phys_context.kinematics.basis.world_to_local(position - self.last_position)

        velocity = self.sim.phys_context.kinematics.basis.world_to_local(self.sim.phys_context.kinematics.world.get_velocity())

        jitter_1st_order, jitter_2nd_order = self.sim.states.sim.joints.get_jitter()
        joint_variance = np.var(self.sim.states.sim.joints.real.deg.get()[:], axis=0)

        imu_gyro = self.sim.phys_context.sensors.imu_gyro.internal.get()
        imu_accel = self.sim.phys_context.sensors.imu_accel.internal.get()

        paw_clearance = self.sim.phys_context.kinematics.foot.paw_clearance()
        paw_slipping, num_paws_contacting = self.sim.phys_context.kinematics.foot.paw_slipping()
        num_arms_contacting = len(self.sim.phys_context.contacts.contacting_geoms(self.sim.phys_context.robot_info.arm_geom_ids))

        roll, pitch = self.sim.phys_context.kinematics.basis.get_tilt()


        return RewardComponents(
            position_delta=position_delta,
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
        

        self.total_distance_traveled += components.position_delta[0]

        movement_reward = self.rewards.WT_Movement * components.position_delta[0]
        movement_penalty = self.rewards.WT_LateralMovement * (abs(components.position_delta[1]**2) + abs(components.position_delta[2]**2))

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


    