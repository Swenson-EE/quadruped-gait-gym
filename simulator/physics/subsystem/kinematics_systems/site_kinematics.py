from simulator.physics.core import PhysicsSubsystem

import mujoco
import numpy as np


class SiteKinematics(PhysicsSubsystem):
    
    def get_position(self, site_id: int) -> np.ndarray:
        return self.data.site_xpos[site_id]

    def get_velocity(self, site_id, local=False):
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_SITE,
            site_id,
            vel,
            int(local)
        )
        return vel[3:]
    
