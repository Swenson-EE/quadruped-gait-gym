from simulator.modules.physics.kinematics import Kinematics
from simulator.modules.physics.kinematics_systems.joint_kinematics import JointKinematics
from simulator.modules.physics_module import Physics
from simulator.modules.reward.reward_subsystem import RewardSubsystem
from simulator.core.registry import register_module
from simulator.modules.reward_module import Reward

import numpy as np


@register_module(Reward)
class JointReward(RewardSubsystem):
    
    def initialize(self):
        self.prev_joint_vel = None
        self.curr_joint_vel = None
        self.next_joint_vel = None

        self._reducers["penalty"]["joint_velocity"] = np.linalg.norm
        self._reducers["penalty"]["joint_acceleration"] = np.linalg.norm
        self._reducers["penalty"]["joint_jerk"] = np.linalg.norm
        self._reducers["penalty"]["joint_energy"] = np.linalg.norm

        self._normalization_factor["penalty"]["joint_velocity"] = 100
        self._normalization_factor["penalty"]["joint_acceleration"] = 10
        self._normalization_factor["penalty"]["joint_jerk"] = 1
        self._normalization_factor["penalty"]["joint_energy"] = 10
        

    def reset_end(self, rng):
        self.prev_joint_vel = None
        self.prev_prev_joint_vel = None
        self.curr_joint_vel = None
        self.next_joint_vel = None

        self.curr_action = None

    def step_end(self, rng, action):
        kn_joints: JointKinematics = self.sim.get(Physics).get(Kinematics).get(JointKinematics)
        
        self.curr_joint_vel = kn_joints.get_velocities()
        self.prev_prev_joint_vel = self.prev_joint_vel
        self.prev_joint_vel = self.next_joint_vel
        self.next_joint_vel = self.curr_joint_vel.copy()

        self.curr_action = action

    def _get_components(self):
        joint_velocity_t0 = self.curr_joint_vel
        joint_velocity_t1 = self.prev_joint_vel if self.prev_joint_vel is not None else 0.0
        joint_velocity_t2 = self.prev_prev_joint_vel if self.prev_prev_joint_vel is not None else 0.0

        reward = None
        penalty = {
            "joint_velocity": joint_velocity_t0**2,
            "joint_acceleration": (joint_velocity_t0 - joint_velocity_t1)**2,
            "joint_jerk": (joint_velocity_t0 - 2*joint_velocity_t1 + joint_velocity_t2)**2,
            "joint_energy": (joint_velocity_t0 * self.curr_action)
        }

        return reward, penalty

