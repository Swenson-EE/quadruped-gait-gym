from simulator.core.subsystem.subsystem import Subsystem
from simulator.core.registry import register_module
from simulator.modules.physics.kinematics import Kinematics

import mujoco
import numpy as np


@register_module(Kinematics)
class SiteKinematics(Subsystem):
    
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
    
