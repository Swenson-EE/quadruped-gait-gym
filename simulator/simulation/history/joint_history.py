import numpy as np

class JointHistoryView:
    def __init__(self, parent, scale=1.0, to_unit=lambda x: x, from_unit=lambda x: x):
        self.parent = parent
        self.scale = scale
        self.to_unit = to_unit
        self.from_unit = from_unit

    def push(self, value):
        value = self.to_unit(value) / self.scale
        self.parent._insert(value)

    def set(self, history):
        history = self.to_unit(history) / self.scale
        self.parent._joint_history = history
        self.parent._history_id = 0
    
    def get(self):
        id = self.parent._history_id
        history = np.roll(self.parent._joint_history, -id, axis=0)
        history = self.from_unit(history * self.scale)
        return history

    def get_index(self, n=0):
        id = (self.parent._history_id - n - 1) % self.parent.length_history
        return self.from_unit(self.parent._joint_history[id] * self.scale)



    # --- Unit Views ---
    @property
    def deg(self):
        return JointHistoryView(
            self.parent,
            scale=self.scale,
            to_unit=lambda x: np.deg2rad(x),
            from_unit=lambda x: np.rad2deg(x)
        )
    
    @property
    def rad(self):
        return JointHistoryView(
            parent=self.parent,
            scale=self.scale,
            to_unit=lambda x: x,
            from_unit=lambda x: x
        )



class JointHistory:
    def __init__(self, length_history: int = 20, num_joints: int = 8, angle_limit: int = 100):
        self.length_history = length_history
        self.num_joints = num_joints
        self.angle_limit = angle_limit

        self._joint_history = None
        self._history_id = 0

        self._internal_view = JointHistoryView(self, scale=1.0)
        self._real_view = JointHistoryView(self, scale=angle_limit)

    def size(self):
        return self.length_history, self.num_joints

    def clear(self):
        self._joint_history = np.zeros((self.length_history, self.num_joints))
        self._history_id = 0

    def _insert(self, joint_angles):
        self._joint_history[self._history_id] = joint_angles
        self._history_id = self._next_id()

    def _next_id(self) -> int:
        return (self._history_id + 1) % self.length_history
    

    @property
    def internal(self):
        return self._internal_view
    
    @property
    def real(self):
        return self._real_view

    # def set_history(self, history):
    #     self._joint_history = history
    #     self._history_id = 0

    # def push_joint_angles(self, joint_angles):
    #     self._joint_history[self._history_id] = joint_angles
    #     self._history_id = (self._history_id + 1) % self.length_history

    # def get_history(self, n=0, units='rad'):
    #     id = (self._history_id - n - 1) % self.length_history

    #     if units == 'rad':
    #         return np.deg2rad(self._joint_history[id])
    #     elif units == 'deg':
    #         return self._joint_history[id]
    #     else:
    #         raise Exception("Unsupported units") # unsupported units
        
    # def get_ordered_history(self):
    #     id = self._history_id
    #     return np.roll(self._joint_history, -id, axis=0)

    def get_jitter(self):
        # joint_angle_t = self.get_history(n=0, units='rad')
        # joint_angle_t1 = self.get_history(n=1, units='rad')
        # joint_angle_t2 = self.get_history(n=2, units='rad')
        joint_angle_t = self.real.rad.get_index(n=0)
        joint_angle_t1 = self.real.rad.get_index(n=1)
        joint_angle_t2 = self.real.rad.get_index(n=2)
        
        jitter_1st_order = joint_angle_t - joint_angle_t1
        jitter_2nd_order = joint_angle_t - 2 * joint_angle_t1 + joint_angle_t2

        return np.abs(jitter_1st_order), np.abs(jitter_2nd_order)
    
    

    