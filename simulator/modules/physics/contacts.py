import mujoco

from simulator.core.subsystem.subsystem import Subsystem
from simulator.core.registry import register_module
from simulator.modules.physics_module import Physics

import numpy as np


@register_module(Physics)
class Contacts(Subsystem):
    """
    Handles all simulation collision and ground interaction
    """

    @property
    def ground_id(self):
        return self.sim.robot_info.ground_id


    def is_geom_in_contact_with_ground(self, geom_id):
        for c in self.data.contact[:self.data.ncon]:
            if self.ground_id in (c.geom1, c.geom2) and geom_id in (c.geom1, c.geom2):
                return True
        return False

    def contacting_geoms(self, geom_ids):
        contacting = np.zeros(4)

        for c in self.data.contact[:self.data.ncon]:
            if self.ground_id in (c.geom1, c.geom2):
                for i in range(len(geom_ids)):
                    
                    contacting[i] = geom_ids[i] in (c.geom1, c.geom2)

        return contacting
        # return {  
        #     g
        #     for c in self.data.contact[:self.data.ncon]
        #     if self.ground_id in (c.geom1, c.geom2)
        #     for g in (c.geom1, c.geom2)
        #     if g in geom_ids
        # }

    def contact_normals(self, geom_id):
        normals = []
        for c in self.data.contact[:self.data.ncon]:
            if geom_id in (c.geom1, c.geom2):
                normals.append(c.frame[:3])  # normal
        return normals
