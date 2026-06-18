from dataclasses import asdict, dataclass
from typing import TypeVar, Generic

import gymnasium as gym
import mujoco
import numpy as np

#from parameters import rewards
from shared.config.environment_parameters import EnvironmentParameters
from shared.rewards.components import RewardComponents, RewardNormalizationFactors, normalize_reward_components
from shared.rewards import RewardWeights
from shared.rewards.rewards import compute_total
from simulator.bittle_sim import BittleParameters, BittleSimulator

#from shared.pickle.pickle_fields import find_unpickable_fields


from simulator.modules import Physics, SimulationState
from simulator.modules.reward_module import Reward

from simulator.modules.physics import Kinematics, Contacts, Sensors, LocomotionMetrics
from simulator.modules.physics.kinematics_systems import BasisKinematics, FootKinematics, JointKinematics, WorldKinematics


# @dataclass
# class EnvironmentParameters:
#     joint_min: int = -120
#     joint_max: int = 120
    
#     joint_delta: int = 25
    
#     phase_rate: float = 1.0

#     length_joint_history: int = 50

#     episode_length: int = 250
#     total_length: int = 1e6

# T = TypeVar("T", bound=EnvironmentParameters)


class BaseBittleEnvironment(gym.Env):
    
    def __init__(self, parameters: EnvironmentParameters = EnvironmentParameters(), weights = RewardWeights()):
        super().__init__()

        self.params = parameters
        self.weights = weights

        bittle_params = BittleParameters(
            model_path="model/bittle_mujoco.xml",
            length_joint_history=self.params.length_joint_history
        )

        self.sim = BittleSimulator(parameters=bittle_params)


        self.JOINT_HISTORY_DIM = (self.sim.params.length_joint_history, self.sim.NUM_JOINTS)
        self.GYRO_DIM = 3
        self.ACCEL_DIM = 3
        self.PAW_CONTACT_DIM = 4

        self.step_count = 0
        self.total_session_step_count = 0


        self.phase = 0
        self.phase_rate = 2 * np.pi * self.params.phase_rate


        self.observation_space = gym.spaces.Dict({
            'joint_history': gym.spaces.Box(-1, 1, shape=self.JOINT_HISTORY_DIM, dtype=np.float32),
            'gyro': gym.spaces.Box(-1, 1, shape=(self.GYRO_DIM,), dtype=np.float32),
            'accel': gym.spaces.Box(-1, 1, shape=(self.ACCEL_DIM,), dtype=np.float32),
            'paw_contact': gym.spaces.Box(0, 1, shape=(self.PAW_CONTACT_DIM,), dtype=np.float32),
            'phase': gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        })

    


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0

        self.phase = 0

        self.sim.reset(self.np_random)        

        observation = self.get_observation()
        info = {}
        info["position"] = self.get_position()

        #print('Observation (reset):', observation)
        
        return observation, info


        

    def step(self, action):
        action = self.decode_action(action)

        action = np.clip(
            action * self.params.joint_delta,
            self.params.joint_min,
            self.params.joint_max
        )
        action = np.deg2rad(action)

        physics = self.sim.get(Physics)
        metrics: LocomotionMetrics = physics.get(LocomotionMetrics)

        start_pos = self.get_position()
        self.sim.step(self.np_random, action)
        end_pos = self.get_position()

        self.phase += self.phase_rate * self.sim.params.control_dt

        
        observation = self.get_observation()

        sys_reward = self.sim.get(Reward)
        reward, penalty = sys_reward.get_reward()

        info = self.get_info()

        info["start_position"] = start_pos
        info["end_position"] = end_pos

        #info['reward'] = reward
        #info['penalty'] = penalty
        #info['observation'] = observation
        #info['components'] = sys_reward.get_components()        
        

        total_reward, reward_sum, penalty_sum = compute_total(self.weights, reward, penalty)

        info["reward_sum"] = reward_sum
        info["penalty_sum"] = penalty_sum


        terminated = False
        truncated = False

        self.step_count += 1
        if self.step_count > self.params.episode_length:
            self.total_session_step_count += self.step_count
            terminated = False
            truncated = True

        elif metrics.is_fallen():
            self.total_session_step_count += self.step_count
            total_reward = 0
            terminated = True
            truncated = False

        # print('Observation (step):', observation)

        return observation, total_reward, terminated, truncated, info
    


    def get_observation(self):      
        physics = self.sim.get(Physics)
        sensors = physics.get(Sensors)
        contacts: Contacts = physics.get(Contacts)

        imu_gyro = sensors.imu_gyro
        imu_accel = sensors.imu_accel

        joint_history = self.sim.get(SimulationState).joints
        paw_contact = contacts.contacting_geoms(self.sim.robot_info.foot_geom_ids)


        phase = np.array([
            np.sin(self.phase),
            np.cos(self.phase)
        ])

        return {
            "joint_history": joint_history.internal.get(),
            "gyro": np.clip(imu_gyro.internal.get(), -1.0, 1.0),
            "accel": np.clip(imu_accel.internal.get(), -1.0, 1.0),
            "paw_contact": paw_contact,
            "phase": phase
        }

    
    def get_info(self):
        return {}
    
    def decode_action(self, action):
        return action


    def get_position(self):
        kn_world = self.sim.get(Physics).get(Kinematics).get(WorldKinematics)
        return kn_world.get_position()

    


