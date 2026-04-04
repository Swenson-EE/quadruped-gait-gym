from simulator.core.transforms import AngleTransformableBuffer

from simulator.core.registry import SubsystemRegistry
from simulator.core.subsystem import ModularSubsystem


@SubsystemRegistry.register
class SimulationState(ModularSubsystem):

    joints = None

    def initialize(self):
        self.joints = AngleTransformableBuffer(
            size=(20, 8),
            scale=100.0
        )

    def reset(self, rng):
        self.joints.clear()

    