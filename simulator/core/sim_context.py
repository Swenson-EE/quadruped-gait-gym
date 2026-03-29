import mujoco
from .robot_info import RobotInfo

from .kinematics import Kinematics
from .contacts import Contacts
from .dynamics import Dynamics
from .sensors import Sensors
from .locomotion_metrics import LocomotionMetrics


class SimulationContext:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.robot_info = RobotInfo(model)


        self.kinematics = Kinematics(self)
        self.contacts = Contacts(self)
        self.dynamics = Dynamics(self)
        self.sensors = Sensors(self)
        self.metrics = LocomotionMetrics(self)

    

