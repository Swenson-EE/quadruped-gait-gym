import mujoco
import numpy as np

from simulator.core.subsystem.subsystem import Subsystem
from simulator.core.registry import register_module
from simulator.modules.physics_module import Physics

@register_module(Physics)
class Dynamics(Subsystem):
    """
    Handles info such as forces and impulses
    """

    def contact_forces(self, contact_index):
        force = np.zeros(6)
        mujoco.mj_contactForce(self.model, self.data, contact_index, force)
        return force
    

