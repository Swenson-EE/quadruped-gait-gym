from simulator.core.subsystem.subsystem import Subsystem
from simulator.core.registry import register_module
from simulator.modules.physics.kinematics import Kinematics


@register_module(Kinematics)
class WorldKinematics(Subsystem):
    
    def get_position(self):
        return self.data.qpos[0:3]
    
    def get_velocity(self):
        return self.data.qvel[0:3]
    

