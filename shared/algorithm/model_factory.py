from stable_baselines3 import DDPG, SAC

class ModelFactory:
    # class: set of kwargs to remove
    BLOCKED_KWARGS = {
        # SAC and DDPG Dict observations don't support n_steps
        SAC: {"n_steps"},
        DDPG: {"n_steps"}
    }

    @staticmethod
    def create(cls, env, **kwargs):
        import inspect

        # Step 1: remove blocked kwargs
        blocked = ModelFactory.BLOCKED_KWARGS.get(cls, set())
        filtered_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in blocked
        }

        # Step 2: normal signature filtering
        sig = inspect.signature(cls.__init__)

        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )

        if not has_var_kw:
            filtered_kwargs = {
                k: v for k, v in filtered_kwargs.items()
                if k in sig.parameters
            }

        return cls("MultiInputPolicy", env, **filtered_kwargs)
    
    