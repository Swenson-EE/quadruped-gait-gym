from simulator.physics.core import PhysicsSubsystem

import mujoco
import numpy as np


class JointKinematics(PhysicsSubsystem):

    @property
    def _joint_qpos_ids(self):
        return self.context.robot_info.joint_qpos_ids

    def get_angles(self, units = 'deg') -> np.ndarray:
        # Get joint angles (rad)
        joint_angles = self.data.qpos[self._joint_qpos_ids]
        if units == 'deg':
            joint_angles = np.rad2deg(joint_angles)
        
        return joint_angles.astype(int)
            
    def set_angles(self, angles: np.ndarray):
        #print('joint qpos:', self.data.qpos[self._joint_qpos_ids])

        #self.data.qpos[self._joint_qpos_ids] = angles
        self.data.qpos[7:15] = angles
        #print('joint qpos:', self.data.qpos[self._joint_qpos_ids])
