import numpy as np

from simulator.core.registry import SubsystemRegistry
from simulator.core.subsystem import ModularSubsystem


@SubsystemRegistry.register
class Randomization(ModularSubsystem):
    pass

