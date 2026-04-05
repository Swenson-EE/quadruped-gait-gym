from simulator.core.subsystem.subsystem import Subsystem
from simulator.core.registry import register_module
from simulator.modules import Randomization

from simulator.modules.physics_module import Physics
from simulator.modules.physics.kinematics import Kinematics
from simulator.modules.physics.kinematics_systems import JointKinematics

from simulator.modules.simulation_state import SimulationState


import numpy as np


@register_module(Randomization, noise_scale=5)
class JointHistoryRandomizer(Subsystem):
    def __init__(self, sim, noise_scale):
        super().__init__(sim)

        self.noise_scale = noise_scale
    

    @property
    def size(self):
        simulation_state = self.sim.get(SimulationState)
        length_history, num_joints = simulation_state.joints.size()

        return length_history, num_joints


    def reset_end(self, rng):
        length_history, num_joints = self.size
        joint_history = np.zeros((length_history, num_joints), dtype=np.float32)
        
        kinematics = self.sim.get(Physics).get(Kinematics)
        kn_joint = kinematics.get(JointKinematics)
        joint_angles = kn_joint.get_angles()
        

        noise = rng.normal(0, self.noise_scale, size=self.size)
        joint_history = joint_angles + noise
        joint_history[0] = joint_angles
        
        simulation_state = self.sim.get(SimulationState)
        simulation_state.joints.real.deg.set(joint_history)


