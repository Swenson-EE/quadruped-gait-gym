import mujoco
import numpy as np

from simulator.physics.core import PhysicsSubsystem

class Dynamics(PhysicsSubsystem):
    """
    Handles info such as forces and impulses
    """

    def contact_forces(self, contact_index):
        force = np.zeros(6)
        mujoco.mj_contactForce(self.model, self.data, contact_index, force)
        return force
    

