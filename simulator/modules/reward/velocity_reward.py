from simulator.modules.reward.reward_subsystem import RewardSubsystem
from simulator.core.registry import register_module
from simulator.modules.reward_module import Reward

from simulator.modules.physics_module import Physics
from simulator.modules.physics.kinematics import Kinematics
from simulator.modules.physics.kinematics_systems.basis_kinematics import BasisKinematics
from simulator.modules.physics.kinematics_systems.world_kinematics import WorldKinematics

import numpy as np


@register_module(Reward)
class VelocityReward(RewardSubsystem):

    initial_rotation = None

    def initialize(self):
        self._reducers['penalty']['lateral_velocity'] = lambda x: x**2
        self._reducers['penalty']['z_velocity'] = lambda x: x**2


    def _get_rotation(self):
        kn_basis = self.sim.get(Physics).get(Kinematics).get(BasisKinematics)
        return kn_basis.rot_mat
    
    def reset_end(self, rng):
        self.initial_rotation = self._get_rotation()


    def _get_components(self):
        physics = self.sim.get(Physics)

        kinematics = physics.get(Kinematics)
        kn_world = kinematics.get(WorldKinematics)

        #velocity = self.initial_rotation.T @ kn_world.get_velocity()
        velocity = kn_world.get_velocity()

        reward = {
            "forward_velocity": velocity[0],
        }
        penalty = {
            "lateral_velocity": abs(velocity[1]),
            "z_velocity": abs(velocity[2])
        }

        return reward, penalty

