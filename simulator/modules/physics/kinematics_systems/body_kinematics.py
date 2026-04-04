from simulator.core.subsystem.subsystem import Subsystem
from simulator.core.registry import register_module
from simulator.modules.physics.kinematics import Kinematics

import mujoco
import numpy as np

@register_module(Kinematics)
class BodyKinematics(Subsystem):
    
    def get_position(self, body_id):
        return self.data.xpos[body_id]
    
    def get_orientation(self, body_id):
        return self.data.xquat[body_id]
    
    def get_velocity(self, body_id, local=False):
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            vel,
            int(local)
        )
        return vel[3:]  # linear

