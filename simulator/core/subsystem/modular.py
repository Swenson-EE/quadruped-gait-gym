from simulator.core.subsystem.node import Node
from simulator.core.registry import ModuleRegistry

class Modular:

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.module_registry = ModuleRegistry()

    def __init__(self):
        self._modules = {}

    def _init_modules(self, owner):
        self._modules = self.module_registry.create_all(owner)

    def get(self, module_cls):
        return self._modules[module_cls]
    
    def initialize(self):
        pass


    def reset(self, rng):
        for module in self._modules.values():
            module.reset(rng)

    def step(self, rng):
        for module in self._modules.values():
            module.step(rng)

