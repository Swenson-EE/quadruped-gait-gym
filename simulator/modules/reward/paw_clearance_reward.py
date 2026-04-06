from simulator.modules.reward.reward_subsystem import RewardSubsystem
from simulator.core.registry import register_module
from simulator.modules.reward_module import Reward

from simulator.modules.physics_module import Physics
from simulator.modules.physics.kinematics import Kinematics
from simulator.modules.physics.kinematics_systems.foot_kinematics import FootKinematics



@register_module(Reward)
class PawClearanceReward(RewardSubsystem):

    PAW_Z_THRESHOLD = 0.005

    def initialize(self):
        self._weight['penalty']['paw_clearance'] = 1
        self._normalization_factor['penalty']['paw_clearance'] = 0.05

        self._reducers['penalty']['paw_clearance'] = lambda x: sum( [max(0, foot_z) for foot_z in x] )


    def _get_components(self):
        
        physics = self.sim.get(Physics)

        kinematics = physics.get(Kinematics)
        kn_foot = kinematics.get(FootKinematics)

        paw_clearance = kn_foot.paw_clearance()

        
        reward = None
        penalty = {
            "paw_clearance": (paw_clearance - self.PAW_Z_THRESHOLD)
        }

        return reward, penalty

