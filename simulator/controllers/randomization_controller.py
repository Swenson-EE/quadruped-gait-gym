import numpy as np

from simulator.core import RandomizationSubsystem


class RandomizationController:
    def __init__(self, modules: list[RandomizationSubsystem]):
        self.modules = modules

    def apply(self, rng: np.random.Generator):
        for module in self.modules:
            module.apply(rng)

