import mujoco
from abc import ABC
import numpy as np

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
        return self.context.systems.kinematics

    @property
    def contacts(self) -> "Contacts":
        return self.context.systems.contacts
    
    @property
    def dynamics(self) -> "Dynamics":
        return self.context.systems.dynamics
    
    @property
    def sensors(self) -> "Sensors":
        return self.context.systems.sensors
    
    @property
    def metrics(self) -> "LocomotionMetrics":
        return self.context.systems.metrics

    def reset(rng: np.random.Generator):
        pass
