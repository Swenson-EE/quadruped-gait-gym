from simulator.core.subsystem.modular import Modular

class Module(Modular):
    def __init__(self, parent, **config):
        super().__init__()

        self.parent = parent
        self.sim = parent.sim if hasattr(parent, "sim") else parent

        self._init_modules(self)
    
    
    