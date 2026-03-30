from simulator.physics.core import PhysicsSubsystem

import mujoco
import numpy as np

class BodyKinematics(PhysicsSubsystem):

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

