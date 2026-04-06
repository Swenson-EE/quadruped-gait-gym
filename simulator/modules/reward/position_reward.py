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
        self._weight['reward']['movement'] = 1
        self._weight['penalty']['movement'] = 100

        self._normalization_factor['reward']['movement'] = 0.01
        self._normalization_factor['penalty']['movement'] = 0.01

        self._reducers['penalty']['movement'] = np.sum


    def _get_rotation(self):
        kn_basis = self.sim.get(Physics).get(Kinematics).get(BasisKinematics)
        return kn_basis.rot_mat

    def _get_position(self):
        kn_world = self.sim.get(Physics).get(Kinematics).get(WorldKinematics)
        return kn_world.get_position().copy()

    def reset_end(self, rng):
        self._initial_rotation = self._get_rotation()

    def step_start(self, rng):
        self.last_position = self._get_position()
        

    def step_end(self, rng):
        #self.position_change = self._initial_rotation.T @ (self._get_position() - self.last_position)
        self.position_change = self._get_position() - self.last_position


    def _get_components(self):
        reward = {
            "movement": self.position_change[0]
        }

        penalty = {
            "movement": np.array(self.position_change[1], self.position_change[2])
        }

        return reward, penalty
    

    

