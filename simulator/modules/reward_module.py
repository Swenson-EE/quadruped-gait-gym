from simulator.core.registry import SubsystemRegistry
from simulator.core.subsystem import ModularSubsystem

from simulator.modules.reward.reward_subsystem import RewardSubsystem

@SubsystemRegistry.register
class Reward(ModularSubsystem):
    def get_reward(self):
        reward, penalty = {}, {}

        for cls, module in self._modules.items():
            if isinstance(module, RewardSubsystem):
                r, p = module.get()
                
                reward.update(r or {})
                penalty.update(p or {})

        return reward, penalty
    
    def get_components(self):
        components = {}

        for cls, module in self._modules.items():
            if isinstance(module, RewardSubsystem):
                r, p = module._get_components()

                components.update(r or {})
                components.update(p or {})

        return components
    
    def get_normalized_components(self):
        components = {}

        for cls, module in self._modules.items():
            if isinstance(module, RewardSubsystem):
                r, p = module.get_normalized()

                components.update(r or {})
                components.update(p or {})

        return components