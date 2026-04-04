
class SubsystemRegistry:
    _registry = {}

    @classmethod
    def register(cls, subsystem_class):
        cls._registry[subsystem_class.__name__] = subsystem_class
        return subsystem_class
    
    @classmethod
    def get_all(cls):
        return cls._registry
    
 