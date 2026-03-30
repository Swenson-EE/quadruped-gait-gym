from abc import ABC, abstractmethod
import numpy as np


class RandomizationSubsystem(ABC):
    def __init__(self, sim: "BittleSimulator"):
        self.sim = sim
        
    @abstractmethod
    def apply(self, rng: np.random.Generator):
        pass

    
    @property
    def context(self):
        return self.sim.context
    
    