from simulator.physics.core import RandomizationSubsystem
import numpy as np

class JointHistoryRandomizer(RandomizationSubsystem):
    def __init__(self, sim, noise=(-15, 15)):
        super().__init__(sim)

        self.noise = noise

    def apply(self, rng):
        length_history, num_joints = self.sim.sim_state.joints.size()
        joint_history = np.empty((length_history, num_joints))
        

        joint_angles = self.context.kinematics.joint.get_angles()
        joint_history[0] = joint_angles

        noise = rng.normal(0, 0.01, size=(length_history - 1, num_joints))
        joint_history[1:] = joint_angles + noise

        self.sim.sim_state.joints.set_history(joint_history.astype(int))

