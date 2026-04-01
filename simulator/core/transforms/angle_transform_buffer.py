from .transformable_buffer import TransformableBuffer, BufferView

import numpy as np

class AngleBufferView(BufferView):
    @property
    def deg(self):
        return BufferView(
            self._parent,
            to_internal=lambda x: np.deg2rad(x),
            from_internal=lambda x: np.rad2deg(x)
        )
    
    @property
    def rad(self):
        return BufferView(
            self._parent,
            to_internal=lambda x: x,
            from_internal=lambda x: x
        )



class AngleTransformableBuffer(TransformableBuffer[AngleBufferView]):
    def __init__(self, size, scale=1.0):
        super().__init__(size, scale, view_cls=AngleBufferView)
    
    def get_jitter(self):
        joint_angle_t = self.real.rad.get_index(n=0)
        joint_angle_t1 = self.real.rad.get_index(n=1)
        joint_angle_t2 = self.real.rad.get_index(n=2)
        
        jitter_1st_order = joint_angle_t - joint_angle_t1
        jitter_2nd_order = joint_angle_t - 2 * joint_angle_t1 + joint_angle_t2

        return np.abs(jitter_1st_order), np.abs(jitter_2nd_order)


