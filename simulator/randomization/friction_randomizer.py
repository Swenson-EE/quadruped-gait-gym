from simulator.physics.core import RandomizationSubsystem
import numpy as np

class FrictionRandomizer(RandomizationSubsystem):
    def __init__(self, sim, f_sliding_range=(0.5, 2.0), f_torsional_range=(0.001, 0.01), f_rotational_range=(0.0001, 0.0002)):
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
        ground_id = self.sim.states.phys.context.robot_info.ground_id
        self.sim.model.geom_friction[ground_id, :] = self.random_friction(rng)

    def _rand_foot_friction(self, rng):
        foot_ids = self.sim.states.phys.context.robot_info.foot_geom_ids
        feet_friction = self.random_friction(rng)

        for id in foot_ids:
            self.sim.model.geom_friction[id, :] = feet_friction


    def apply(self, rng):
        self._rand_ground_friction(rng)
        self._rand_foot_friction(rng)






