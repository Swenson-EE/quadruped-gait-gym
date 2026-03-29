from abc import ABC, abstractmethod
import numpy as np

from .subsystem import RobotSubsystem


class RandomizationSubsystem(RobotSubsystem, ABC):
    def __init__(self, context: "SimulationContext"):
        super().__init__(context)
        
    @abstractmethod
    def apply(self, rng: np.random.Generator):
        pass
    