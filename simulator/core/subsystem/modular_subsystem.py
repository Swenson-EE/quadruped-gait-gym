from simulator.core.subsystem import Subsystem, Modular, Node
from simulator.core.registry import ModuleRegistry

class ModularSubsystem(Modular, Subsystem):
    module_registry = ModuleRegistry()

    def __init__(self, sim):
        Subsystem.__init__(self, sim)
        Modular.__init__(self)
        self._init_modules(self)

    
    
            