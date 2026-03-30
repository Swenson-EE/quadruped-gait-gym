from abc import ABC, abstractmethod
import numpy as np

from .subsystem import PhysicsSubsystem


class RandomizationSubsystem(PhysicsSubsystem, ABC):
    def __init__(self, context: "SimulationContext"):
        super().__init__(context)
        
    @abstractmethod
    def apply(self, rng: np.random.Generator):
        pass
    