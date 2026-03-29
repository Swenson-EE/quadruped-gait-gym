import mujoco
from .robot_info import RobotInfo

from ..subsystems.kinematics import Kinematics
from ..subsystems.contacts import Contacts
from ..subsystems.dynamics import Dynamics
from ..subsystems.sensors import Sensors
from ..subsystems.locomotion_metrics import LocomotionMetrics


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

    

