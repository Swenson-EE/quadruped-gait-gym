import mujoco
from dataclasses import dataclass
from .robot_info import RobotInfo

from simulator.physics.subsystem import Kinematics, Contacts, Dynamics, Sensors, LocomotionMetrics


class PhysicsContext:
    @dataclass
    class Systems:
        kinematics: Kinematics
        contacts: Contacts
        dynamics: Dynamics
        sensors: Sensors
        metrics: LocomotionMetrics

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.robot_info = RobotInfo(model)

        self.systems = PhysicsContext.Systems(
            kinematics=Kinematics(self),
            contacts=Contacts(self),
            dynamics=Dynamics(self),
            sensors=Sensors(self),
            metrics=LocomotionMetrics(self)
        )
        
        