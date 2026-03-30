import numpy as np

class JointHistory:
    def __init__(self, length_history: int = 20, num_joints: int = 8):
        self.length_history = 20
        self.num_joints = 8

        self._joint_history = None
        self._history_id = 0

    def clear(self):
        self._joint_history = np.zeros((self.length_history, self.num_joints))
        self._history_id = 0

    def push_joint_angles(self, joint_angles):
        self._joint_history[self._history_id] = joint_angles
        self._history_id = (self._history_id + 1) & self.length_history

    def get_history(self, n=0, units='rad'):
        id = (self._history_id - n - 1) % self.length_history

        if units == 'rad':
            return np.deg2rad(self._joint_history[id])
        elif units == 'deg':
            return self._joint_history[id]
        else:
            raise Exception("Unsupported units") # unsupported units

    def get_jitter(self):
        joint_angle_t = self.get_history(n=0, units='rad')
        joint_angle_t1 = self.get_history(n=1, units='rad')
        joint_angle_t2 = self.get_history(n=2, units='rad')
        
        jitter_1st_order = joint_angle_t - joint_angle_t1
        jitter_2nd_order = joint_angle_t - 2 * joint_angle_t1 + joint_angle_t2

        return np.abs(jitter_1st_order), np.abs(jitter_2nd_order)
    
    def get_ordered_history(self):
        id = self._history_id
        return np.roll(self._joint_history, -id, axis=0)
