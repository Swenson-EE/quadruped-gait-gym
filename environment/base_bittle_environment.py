from dataclasses import asdict, dataclass
from typing import TypeVar, Generic

import gymnasium as gym
import numpy as np

#from parameters import rewards
from shared.rewards.components import RewardComponents, RewardNormalizationFactors, normalize_reward_components
from shared.rewards.rewards import Rewards
from simulator.bittle_sim import BittleParameters, BittleSimulator

#from shared.pickle.pickle_fields import find_unpickable_fields


from simulator.modules import Physics, SimulationState
from simulator.modules.reward_module import Reward

from simulator.modules.physics import Kinematics, Contacts, Sensors, LocomotionMetrics
from simulator.modules.physics.kinematics_systems import BasisKinematics, FootKinematics, JointKinematics, WorldKinematics


@dataclass
class EnvironmentParameters:
    #length_joint_history: int = 20

    #joint_min: int = -120
    #joint_max: int = 120

    episode_length: int = 250
    total_length: int = 1e6



T = TypeVar("T", bound=EnvironmentParameters)


class BaseBittleEnvironment(gym.Env, Generic[T]):

    def __init__(self, parameters: T = EnvironmentParameters(), weights = {}):
        super().__init__()

        self.params = parameters
        self.weights = weights

        #if self.params.rewards is None:
        #    self.params.rewards = Rewards.from_json_file('config/rewards.json')
        

        bittle_params = BittleParameters(
            model_path="model/bittle_mujoco.xml"
        )

        self.sim = BittleSimulator(parameters=bittle_params)
        self.OBSERVATION_DIM = (self.sim.params.length_joint_history * self.sim.NUM_JOINTS) + self.sim.NUM_IMU_OBS

        self.step_count = 0
        self.total_session_step_count = 0
    


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0

        self.sim.reset(self.np_random)        

        observation = self.get_observation()
        info = {}
        info["position"] = self.get_position()

        #print('Observation (reset):', observation)
        
        return observation, info


        

    def step(self, action):
        decoded_action = self.decode_action(action)

        physics = self.sim.get(Physics)
        metrics = physics.get(LocomotionMetrics)

        start_pos = self.get_position()
        self.sim.step(self.np_random, decoded_action)
        end_pos = self.get_position()
        
        observation = self.get_observation()

        sys_reward = self.sim.get(Reward)
        reward, penalty = sys_reward.get_reward()

        info = self.get_info()

        info["start_position"] = start_pos
        info["end_position"] = end_pos

        info['reward'] = reward
        info['penalty'] = penalty
        info['observation'] = observation
        info['components'] = sys_reward.get_components()

        def print_comp(name):
            print('-'*10, name, '-'*10)
            if name in info['components']:
                print(info['components'][name])

                if name in penalty:
                    print('penalty:', penalty[name])
                elif name in reward:
                    print('reward:', reward[name])
                print('\n'*2)

        #print_comp("joint_velocity")
        #print_comp("joint_acceleration")
        #print_comp("joint_jerk")
        #print_comp("joint_energy")

        #print_comp("forward_movement")
        #print_comp("bent_joint")

        reward_total = 0
        if "reward" in self.weights:
            for k, v in reward.items():
                reward_total += (self.weights["reward"].get(k, 0.0) * v)

        penalty_total = 0
        if "penalty" in self.weights:
            for k, v in penalty.items():
                penalty_total -= (self.weights["penalty"].get(k, 0.0) * v)


        reward_scale = self.weights.get("reward_scale", 1.0)
        penalty_scale = self.weights.get("penalty_scale", 1.0)


        total_reward = (reward_total * reward_scale) - (penalty_total * penalty_scale)

        # print('\n'*2, '-'*10)
        # print(reward)
        # print(penalty)
        
        # print('\n'*2, '-'*10)
        # print(info['components'])

        #total_reward = sum(reward.values()) - sum(penalty.values())
    

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

        imu_gyro = sensors.imu_gyro
        imu_accel = sensors.imu_accel

        joint_history = self.sim.get(SimulationState).joints

        return {
            "joint_history": joint_history.internal.get(),
            "gyro": np.clip(imu_gyro.internal.get(), -1.0, 1.0),
            "accel": np.clip(imu_accel.internal.get(), -1.0, 1.0)
        }

    
    def get_info(self):
        return {}
    
    def decode_action(self, action):
        return action


    def get_position(self):
        kn_world = self.sim.get(Physics).get(Kinematics).get(WorldKinematics)
        return kn_world.get_position()

    