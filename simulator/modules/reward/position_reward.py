from simulator.modules.reward.reward_subsystem import RewardSubsystem
from simulator.core.registry import register_module
from simulator.modules.reward_module import Reward

from simulator.modules.physics.kinematics import Kinematics
from simulator.modules.physics.kinematics_systems.basis_kinematics import BasisKinematics
from simulator.modules.physics.kinematics_systems.world_kinematics import WorldKinematics
from simulator.modules.physics_module import Physics


import numpy as np


@register_module(Reward)
class PositionReward(RewardSubsystem):

    last_position = None
    initial_rotation = None

    def initialize(self):
        #self.total_distance = np.zeros(shape=(3,))

        self._normalization_factor['reward']['forward_movement'] = 0.01
        self._normalization_factor['penalty']['lateral_movement'] = 0.01
        self._normalization_factor['penalty']['z_movement'] = 0.01

    


    def _get_rotation(self):
        kn_basis = self.sim.get(Physics).get(Kinematics).get(BasisKinematics)
        return kn_basis.rot_mat

    def _get_position(self):
        kn_world = self.sim.get(Physics).get(Kinematics).get(WorldKinematics)
        return kn_world.get_position().copy()

    def reset_start(self, rng):
        self.total_distance = np.zeros(shape=(3,))

    def reset_end(self, rng):
        self._initial_rotation = self._get_rotation()

    def step_start(self, rng, action):
        self.last_position = self._get_position()
        

    def step_end(self, rng, action):
        #self.position_change = self._initial_rotation.T @ (self._get_position() - self.last_position)
        self.position_change = self._get_position() - self.last_position

        # forward should be always forward, but lateral and z should be total for penalties
        #self.total_distance += [self.position_change[0], abs(self.position_change[1]), abs(self.position_change[2])]



    def _get_components(self):
        reward = {
            #"forward_movement": self.dx,
            "efficiency": self.dx / (abs(self.dy) + abs(self.dz) + 1e-6)
        }

        penalty = {
            "lateral_movement": abs(self.dy),
            "z_movement": abs(self.dz)
        }

        return reward, penalty
    

    @property
    def dx(self):
        return self.position_change[0] 

    @property
    def dy(self):
        return self.position_change[1]
    
    @property
    def dz(self):
        return self.position_change[2]


