from simulator.core.subsystem.subsystem import Subsystem
from simulator.core.registry import register_module

from simulator.modules.physics_module import Physics
from simulator.modules.physics.kinematics import Kinematics
from simulator.modules.physics.contacts import Contacts
from simulator.modules.physics.kinematics_systems.site_kinematics import SiteKinematics
from simulator.modules.physics.kinematics_systems.body_kinematics import BodyKinematics

import mujoco
import numpy as np


@register_module(Kinematics)
class FootKinematics(Subsystem):
    def paw_clearance(self):
        paw_site_ids = self.sim.robot_info.foot_site_ids
        kn_site = self.sim.get(Physics).get(Kinematics).get(SiteKinematics)

        clearances = []
        ray_dir = np.array([0, 0, -1.0])

        for site_id in paw_site_ids:
            paw_pos = kn_site.get_position(site_id)

            geomid = np.array([-1], dtype=np.int32) # geomid[0] -> id of hit geom
            normal = np.zeros(3) # surface normal at hit

            distance = mujoco.mj_ray(
                self.model, 
                self.data, 
                paw_pos,    # ray start
                ray_dir,
                None,       # geomgroup
                1,          # include static geoms
                0,          # exclude no body
                geomid,
                normal
            )

            clearance = paw_pos + (distance * ray_dir)
            clearances.append(clearance[2])

        return np.array(clearances)
    

    def paw_slipping(self):
        slipping = 0.0
        num_paws_contacting = 0
        paw_geom_ids = self.sim.robot_info.foot_geom_ids

        physics = self.sim.get(Physics)

        contacts = physics.get(Contacts)
        kinematics = physics.get(Kinematics)
        
        kn_body = kinematics.get(BodyKinematics)


        for geom_id in contacts.contacting_geoms(paw_geom_ids):
            num_paws_contacting += 1
            body_id = self.model.geom_bodyid[geom_id]
            vel = kn_body.get_velocity(body_id)


            normals = contacts.contact_normals(geom_id)
            for n in normals:
                vel = vel - np.dot(vel, n) * n

            slipping += np.linalg.norm(vel)
        
        return slipping, num_paws_contacting

