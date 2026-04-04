
class ModuleRegistry:
    def __init__(self):
        self._modules = []

    def register(self, module_class, config=None):
        #self._modules[module_class] = module_class
        #return module_class
        self._modules.append((module_class, config or {}))
    
    def create_all(self, owner):
        instances = {}

        for module_class, config in self._modules:
            instances[module_class] = module_class(owner, **config)

        for instance in instances.values():
            instance.initialize()
        
        return instances
    
    