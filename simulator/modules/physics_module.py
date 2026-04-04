from simulator.core.registry import SubsystemRegistry
from simulator.core.subsystem import ModularSubsystem


@SubsystemRegistry.register
class Physics(ModularSubsystem):
    pass
