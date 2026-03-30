from dataclasses import dataclass, field

import mujoco
import numpy as np

from simulator.physics.state import PhysicsState
from simulator.simulation import SimulationState
from simulator.controllers import RandomizationController
import simulator.randomization as s_rnd


@dataclass
class BittleParameters:
    model_path: str = ''

    control_dt = 0.01


class BittleSimulator:
    """
    Simulator class for the Bittle quadruped robotic dog
    """

    NUM_JOINTS = 8          # Number of joints for quadruped
    NUM_IMU_OBS = 6         # Number of observations from IMU (Gyro: 3, Accel: 3)

    def __init__(self, parameters: BittleParameters = BittleParameters()):
        self.params = parameters

        # Initialize the Mujoco simulation
        self.model = mujoco.MjModel.from_xml_path(parameters.model_path)
        self.data = mujoco.MjData(self.model)

        self.n_substeps = int(parameters.control_dt / self.model.opt.timestep)      # The number of substeps to take in a single physics step to simulate control delay

        self.phys_state = PhysicsState(self.model, self.data)
        self.sim_state = SimulationState(self.model, self.data)

        self.randomization = RandomizationController(modules=[
            s_rnd.InitialPoseRandomizer(
                sim=self
            ),
            s_rnd.JointRandomizer(
                sim=self,
                joint_qpos_ids=self.context.robot_info.joint_qpos_ids
            )
        ])

        
    @property
    def context(self):
        return self.phys_state.context

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.context.kinematics.basis.update_rotation()

    def step(self, action = None):
        if action is not None:
            self.data.ctrl[:] = action

        for _ in range(self.n_substeps): # Simulate control updates
            mujoco.mj_step(self.model, self.data)

        self.context.kinematics.basis.update_rotation()

    def forward(self):
        mujoco.mj_forward(self.model, self.data)
    

