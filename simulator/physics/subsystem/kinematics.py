from dataclasses import field
import numpy as np

from simulator.physics.core import PhysicsSubsystem
import simulator.physics.subsystem.kinematics_systems as ks


class Kinematics(PhysicsSubsystem):
    """
    Handles info such as positions, orientations, and velocities
    """

    def __init__(self, context):
        super().__init__(context)

        self.world = ks.WorldKinematics(self.context)
        self.body = ks.BodyKinematics(self.context)
        self.site = ks.SiteKinematics(self.context)

        self.joint = ks.JointKinematics(self.context)
        self.foot = ks.FootKinematics(self.context)

        self.basis = ks.BasisKinematics(self.context) 

