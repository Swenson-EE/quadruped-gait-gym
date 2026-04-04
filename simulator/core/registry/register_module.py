
def register_module(parent_cls, **config):
    def decorator(module_cls):
        parent_cls.module_registry.register(module_cls, config)
        return module_cls
    return decorator