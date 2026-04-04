import mujoco
import numpy as np

from simulator.core.subsystem.subsystem import Subsystem
from simulator.core.registry import register_module
from simulator.modules.physics_module import Physics
from simulator.modules.physics.kinematics import Kinematics

from simulator.modules.physics.kinematics_systems.world_kinematics import WorldKinematics
from simulator.modules.physics.kinematics_systems.basis_kinematics import BasisKinematics


@register_module(Physics)
class LocomotionMetrics(Subsystem):    
    
    def is_moving(self, moving_threshold = 1e-3):
        kinematics = self.sim.get(Physics).get(Kinematics)
        kn_world = kinematics.get(Kinematics).get(WorldKinematics)
        kn_basis = kinematics.get(Kinematics).get(BasisKinematics)

        velocity = kn_world.get_velocity()
        local_velocity = kn_basis.world_to_local(velocity)
        return np.linalg.norm(local_velocity) > moving_threshold
    
    def is_fallen(self, fall_threshold = 1.3):
        kinematics = self.sim.get(Physics).get(Kinematics)
        kn_basis = kinematics.get(BasisKinematics)

        roll, pitch = kn_basis.get_tilt()
        is_fallen = (np.fabs(roll) > fall_threshold) or (np.fabs(pitch) > fall_threshold)
        return is_fallen

