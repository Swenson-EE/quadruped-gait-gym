from simulator.physics.core import PhysicsSubsystem

class WorldKinematics(PhysicsSubsystem):
    
    def get_position(self):
        return self.data.qpos[0:3]
    
    def get_velocity(self):
        return self.data.qvel[0:3]
    
    
    

