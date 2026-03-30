import mujoco
from abc import ABC

class PhysicsSubsystem(ABC):
    def __init__(self, context: "SimulationContext"):
        self.context = context

    @property
    def model(self):
        return self.context.model
    
    @property
    def data(self):
        return self.context.data

    @property
    def info(self):
        return self.context.robot_info
    

    @property
    def kinematics(self) -> "Kinematics":
        return self.context.kinematics

    @property
    def contacts(self) -> "Contacts":
        return self.context.contacts
    
    @property
    def dynamics(self) -> "Dynamics":
        return self.context.dynamics
    
    @property
    def sensors(self) -> "Sensors":
        return self.context.sensors
    
    @property
    def metrics(self) -> "LocomotionMetrics":
        return self.context.metrics
