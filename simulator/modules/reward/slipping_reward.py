from simulator.modules.reward.reward_subsystem import RewardSubsystem
from simulator.core.registry import register_module
from simulator.modules.reward_module import Reward

from simulator.modules.physics_module import Physics
from simulator.modules.physics.kinematics import Kinematics
from simulator.modules.physics.kinematics_systems.foot_kinematics import FootKinematics


#@register_module(Reward)
class SlippingReward(RewardSubsystem):

    def initialize(self):
        pass

    def _get_components(self):
        
        physics = self.sim.get(Physics)

        kinematics = physics.get(Kinematics)
        kn_foot = kinematics.get(FootKinematics)

        paw_slipping, paw_contacts = kn_foot.paw_slipping()

        reward = None
        penalty = {
            "slipping": paw_slipping,
            "paws_contacting": float(len(paw_contacts) < 2)
        }

        return reward, penalty
