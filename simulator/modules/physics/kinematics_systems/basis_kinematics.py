from simulator.core.subsystem.subsystem import Subsystem
from simulator.core.registry import register_module
from simulator.modules.physics.kinematics import Kinematics

from dataclasses import field
import numpy as np
import mujoco


@register_module(Kinematics)
class BasisKinematics(Subsystem):

    _rot_mat: np.ndarray = field(init=False, default=None)

    def get_quat(self):
        return self.data.qpos[3:7] # w, x, y, z

    def update_rotation(self):
        quat = self.get_quat()
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, quat)
        self._rot_mat = rot_mat.reshape(3, 3)

    @property
    def rot_mat(self):
        if self._rot_mat is None:
            self.update_rotation()
        return self._rot_mat
    
    def world_to_local(self, vec: np.ndarray) -> np.ndarray:
        return self.rot_mat.T @ vec # world -> local
    
    def local_to_world(self, vec: np.ndarray) -> np.ndarray:
        return self.rot_mat @ vec # local -> world

    def get_tilt(self):
        """Returns (roll, pitch) in radians from the base quaternion."""
        # Quaternion components
        qw, qx, qy, qz = self.get_quat()

        # Roll (x-axis)
        roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))

        # Pitch (y-axis)
        pitch = np.arctan2(2*(qw*qy - qz*qx), np.sqrt(1 - (2*(qw*qy - qz*qx))**2))
        
        return roll, pitch


    def reset_end(self, rng):
        self.update_rotation()

    def step_start(self, rng):
        self.update_rotation()
