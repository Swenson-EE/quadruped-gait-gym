import mujoco
import numpy as np

from simulator.physics.core import PhysicsSubsystem


class LocomotionMetrics(PhysicsSubsystem):    
    
    def is_moving(self, moving_threshold = 1e-3):
        velocity = self.kinematics.world.get_velocity()
        local_velocity = self.kinematics.basic.world_to_local(velocity)
        return np.linalg.norm(local_velocity) > moving_threshold
    
    def is_fallen(self, fall_threshold = 1.3):
        roll, pitch = self.kinematics.basis.get_tilt()
        is_fallen = (np.fabs(roll) > fall_threshold) or (np.fabs(pitch) > fall_threshold)
        return is_fallen

