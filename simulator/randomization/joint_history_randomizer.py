from simulator.physics.core import RandomizationSubsystem
import numpy as np

class JointHistoryRandomizer(RandomizationSubsystem):
    def __init__(self, sim, noise_scale=5):
        super().__init__(sim)

        self.noise_scale = noise_scale

    @property
    def size(self):
        length_history, num_joints = self.sim.states.sim.joints.size()
        return length_history, num_joints


    def apply(self, rng):
        #length_history, num_joints = self.size
        joint_history = np.zeros((20, 8), dtype=np.float32)
        
        joint_angles = self.context.kinematics.joint.get_angles()
        

        noise = rng.normal(0, self.noise_scale, size=self.size)
        joint_history = joint_angles + noise
        joint_history[0] = joint_angles
        
        self.sim.states.sim.joints.real.deg.set(joint_history)

