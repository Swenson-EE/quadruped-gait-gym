import mujoco

from ..core.subsystem import RobotSubsystem


class Contacts(RobotSubsystem):
    """
    Handles all simulation collision and ground interaction
    """

    @property
    def ground_id(self):
        return self.context.robot_info.ground_id


    def geom_in_contact_with_ground(self, geom_id):
        for c in self.data.contact[:self.data.ncon]:
            if self.ground_id in (c.geom1, c.geom2) and geom_id in (c.geom1, c.geom2):
                return True
        return False

    def contacting_geoms(self, geom_ids):
        return [
            g
            for c in self.data.contact[:self.data.ncon]
            if self.ground_id in (c.geom1, c.geom2)
            for g in (c.geom1, c.geom2)
            if g in geom_ids
        ]

    def contact_normals(self, geom_id):
        normals = []
        for c in self.data.contact[:self.data.ncon]:
            if geom_id in (c.geom1, c.geom2):
                normals.append(c.frame[:3])  # normal
        return normals
