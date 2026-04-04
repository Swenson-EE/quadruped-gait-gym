from simulator.core.subsystem.subsystem import Subsystem
from simulator.core.registry import register_module
from simulator.modules import Randomization

import numpy as np


@register_module(Randomization, f_sliding_range=(0.5, 2.0), f_torsional_range=(0.001, 0.01), f_rotational_range=(0.0001, 0.0002))
class FrictionRandomizer(Subsystem):
    def __init__(self, sim, f_sliding_range, f_torsional_range, f_rotational_range):
        super().__init__(sim)

        self.sliding_friction_range = f_sliding_range
        self.torsional_friction_range = f_torsional_range
        self.rotational_friction_raange = f_rotational_range
    
    

    def random_friction(self, rng):
        sliding_friction = rng.uniform(*self.sliding_friction_range)
        torsional_friction = rng.uniform(*self.torsional_friction_range)
        rotational_friction = rng.uniform(*self.rotational_friction_raange)
        return [sliding_friction, torsional_friction, rotational_friction]
    
    def _rand_ground_friction(self, rng):
        self.model.geom_friction[self.sim.robot_info.ground_id, :] = self.random_friction(rng)

    def _rand_foot_friction(self, rng):
        robot_info = self.sim.robot_info
        feet_friction = self.random_friction(rng)

        for id in robot_info.foot_geom_ids:
            self.model.geom_friction[id, :] = feet_friction


    def reset(self, rng):
        self._rand_ground_friction(rng)
        self._rand_foot_friction(rng)





