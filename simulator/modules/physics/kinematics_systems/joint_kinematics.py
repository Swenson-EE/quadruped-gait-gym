from simulator.core.subsystem.subsystem import Subsystem
from simulator.core.registry import register_module
from simulator.modules.physics.kinematics import Kinematics

import mujoco
import numpy as np


@register_module(Kinematics)
class JointKinematics(Subsystem):

    @property
    def _joint_qpos_ids(self):
        return self.sim.robot_info.joint_qpos_ids

    def get_angles(self, units = 'deg') -> np.ndarray:
        # Get joint angles (rad)
        #joint_angles = self.data.qpos[self._joint_qpos_ids]
        joint_angles = self.data.qpos[7:15]
        
        if units == 'deg':
            joint_angles = np.rad2deg(joint_angles)
        
        return joint_angles.astype(int)
            
    def set_angles(self, angles: np.ndarray):
        self.data.qpos[7:15] = angles

    def get_velocities(self):
        return self.data.qvel[7:15]
