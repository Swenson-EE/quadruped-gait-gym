from simulator.core.subsystem.subsystem import Subsystem
from simulator.core.registry import register_module
from simulator.modules.physics.kinematics import Kinematics


@register_module(Kinematics)
class WorldKinematics(Subsystem):
    
    _POSITION_CHANGE = [-1, 1, 1] # Change the x-axis

    def get_position(self):
        pos = self.data.qpos[0:3].copy()
        pos[0] = -pos[0]
        return pos
    
    def get_velocity(self):
        vel = self.data.qvel[0:3].copy()
        vel[0] = -vel[0]
        return vel
    

