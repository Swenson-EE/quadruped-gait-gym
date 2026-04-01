import mujoco
from .robot_info import RobotInfo

from simulator.physics.subsystem import Kinematics, Contacts, Dynamics, Sensors, LocomotionMetrics


class PhysicsContext:

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.robot_info = RobotInfo(model)

        self.kinematics = Kinematics(self)
        self.contacts = Contacts(self)
        self.dynamics = Dynamics(self)
        self.sensors = Sensors(self)
        self.metrics = LocomotionMetrics(self)
        
        