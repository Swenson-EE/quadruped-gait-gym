from simulator.modules.physics.contacts import Contacts
from simulator.modules.reward.reward_subsystem import RewardSubsystem
from simulator.core.registry import register_module
from simulator.modules.reward_module import Reward

from simulator.modules.physics_module import Physics


@register_module(Reward)
class CrawlReward(RewardSubsystem):
    
    def initialize(self):
        self._normalization_factor['penalty']['crawling'] = 1.0
        

    def _get_components(self):
        physics = self.sim.get(Physics)
        contacts = physics.get(Contacts)

        num_arms_contacting = len(contacts.contacting_geoms(self.sim.robot_info.arm_geom_ids))
        
        reward = None
        penalty = {
            "crawling": num_arms_contacting
        }

        return reward, penalty
