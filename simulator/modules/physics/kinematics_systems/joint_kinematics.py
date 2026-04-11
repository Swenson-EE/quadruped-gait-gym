from simulator.core.subsystem.subsystem import Subsystem
from simulator.core.registry import register_module
from simulator.modules.physics.kinematics import Kinematics

import mujoco
import numpy as np


@register_module(Kinematics)
class JointKinematics(Subsystem):

    @property
    def _joint_qpos_ids(self):
        print(self.sim.robot_info.joint_qpos_ids)
        return self.sim.robot_info.joint_qpos_ids

    def get_angles(self, units = 'deg', j_type=np.float32) -> np.ndarray:
        # Get joint angles (rad)
        joint_angles = self.data.qpos[self.sim.robot_info.joint.qpos_addr]
        
        if units == 'deg':
            joint_angles = np.rad2deg(joint_angles)
        
        return joint_angles.astype(j_type)
            
    def set_angles(self, angles: np.ndarray):
        self.data.qpos[self.sim.robot_info.joint.qpos_addr] = angles

    def get_velocities(self):
        return self.data.qvel[self.sim.robot_info.joint.qvel_addr]
