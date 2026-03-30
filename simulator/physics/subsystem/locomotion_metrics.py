import mujoco
import numpy as np

from simulator.physics.core import PhysicsSubsystem


class LocomotionMetrics(PhysicsSubsystem):


    def paw_clearance(self):
        paw_site_ids = self.context.robot_info.foot_site_ids

        clearances = []
        ray_dir = np.array([0, 0, -1.0])

        for site_id in paw_site_ids:
            paw_pos = self.context.kinematics.site_position(site_id)

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
        paw_geom_ids = self.context.robot_info.foot_geom_ids

        for geom_id in self.contacts.contacting_geoms(paw_geom_ids):
            num_paws_contacting += 1
            body_id = self.kinematics.model.geom_bodyid[geom_id]
            vel = self.context.kinematics.body_velocity(body_id)

            normals = self.context.contacts.contact_normals(geom_id)
            for n in normals:
                vel = vel - np.dot(vel, n) * n

            slipping += np.linalg.norm(vel)

        return slipping, num_paws_contacting
    
    
    def is_moving(self, moving_threshold = 1e-3):
        velocity = self.kinematics.get_velocity()
        local_velocity = self.kinematics.world_to_local(velocity)
        return np.linalg.norm(local_velocity) > moving_threshold
    
    def is_fallen(self, fall_threshold = 1.3):
        roll, pitch = self.kinematics.get_tilt()
        is_fallen = (np.fabs(roll) > fall_threshold) or (np.fabs(pitch) > fall_threshold)
        return is_fallen

