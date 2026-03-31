from simulator.physics.core import RandomizationSubsystem
import numpy as np

class JointHistoryRandomizer(RandomizationSubsystem):
    def __init__(self, sim, noise_scale=5):
        super().__init__(sim)

        self.noise_scale = noise_scale

    @property
    def size(self):
        length_history, num_joints = self.sim.states.sim_state.joints.size()
        return length_history, num_joints


    def apply(self, rng):
        length_history, num_joints = self.size
        joint_history = np.zeros((length_history, num_joints), dtype=np.float32)
        
        joint_angles = self.context.systems.kinematics.joint.get_angles()
        joint_history[0] = joint_angles

        noise = rng.normal(0, self.noise_scale, size=(length_history - 1, num_joints))
        joint_history[1:] = joint_angles + noise

        self.sim.states.sim_state.joints.set_history(joint_history)

