from dataclasses import field

import mujoco
from mujoco import MjModel, MjData

import numpy as np

from .subsystem import RobotSubsystem


class Kinematics(RobotSubsystem):
    """
    Handles info such as positions, orientations, and velocities
    """

    _rot_mat: np.ndarray = field(init=False, default=None)


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
        return self.rot_mat @ vec


    def get_position(self):
        return self.data.qpos[0:3]
    
    def get_velocity(self):
        return self.data.qvel[0:3]
    
    def get_quat(self):
        return self.data.qpos[3:7] # w, x, y, z


    def body_position(self, body_id):
        return self.data.xpos[body_id]
    
    def body_orientation(self, body_id):
        return self.data.xquat[body_id]
    
    def body_velocity(self, body_id, local=False):
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            vel,
            int(local)
        )
        return vel[3:]  # linear
    

    def site_position(self, site_id: int) -> np.ndarray:
        return self.data.site_xpos[site_id]

    def site_velocity(self, site_id, local=False):
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_SITE,
            site_id,
            vel,
            int(local)
        )
        return vel[3:]


    def get_tilt(self):
        """Returns (roll, pitch) in radians from the base quaternion."""
        # Quaternion components
        #qx, qy, qz, qw = self.data.qpos[3:7][1], self.data.qpos[3:7][2], self.data.qpos[3:7][3], self.data.qpos[3:7][0]
        qw, qx, qy, qz = self.get_quat()

        # Roll (x-axis)
        roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))

        # Pitch (y-axis)
        pitch = np.arctan2(2*(qw*qy - qz*qx), np.sqrt(1 - (2*(qw*qy - qz*qx))**2))
        
        return roll, pitch
