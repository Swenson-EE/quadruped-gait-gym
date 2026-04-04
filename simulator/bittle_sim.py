from dataclasses import dataclass, field

import mujoco
import numpy as np

from shared.utils.auto_import import auto_import
auto_import('simulator.modules')

from simulator.core.robot_info import RobotInfo

from simulator.core.registry import SubsystemRegistry





@dataclass
class BittleParameters:
    model_path: str = ''

    control_dt = 0.01

    length_joint_history: int = 20
    joint_max: int = 100


class BittleSimulator:
    """
    Simulator class for the Bittle quadruped robotic dog
    """

    # @dataclass
    # class SimulatorStates:
    #     def __init__(self, model, data):
    #         self.phys = PhysicsState(model, data)
    #         self.sim = SimulationState(model, data)

            


    NUM_JOINTS = 8          # Number of joints for quadruped
    NUM_IMU_OBS = 6         # Number of observations from IMU (Gyro: 3, Accel: 3)

    def __init__(self, parameters: BittleParameters = BittleParameters()):
        self.params = parameters

        # Initialize the Mujoco simulation
        self.model = mujoco.MjModel.from_xml_path(parameters.model_path)
        self.data = mujoco.MjData(self.model)

        self.n_substeps = int(parameters.control_dt / self.model.opt.timestep)      # The number of substeps to take in a single physics step to simulate control delay
        
        self.robot_info = RobotInfo(self.model)


        self._systems = {}

        for name, cls in SubsystemRegistry.get_all().items():
            instance = cls(self)
            instance.initialize()
            self._systems[cls] = instance
            
            #print('sim subsystem:', cls)
            #print('sim subsystem instance:', instance)


        #self.states = BittleSimulator.SimulatorStates(self.model, self.data)

        # self.randomization = RandomizationController(modules=[
        #     s_rnd.InitialPoseRandomizer(
        #         sim=self
        #     ),
        #     s_rnd.JointRandomizer(
        #         sim=self,
        #         joint_qpos_ids=self.phys_context.robot_info.joint_qpos_ids
        #     ),
        #     s_rnd.JointHistoryRandomizer(
        #         sim=self
        #     ),
        #     s_rnd.FrictionRandomizer(
        #         sim=self
        #     )
        # ])

        self.options = {
            'bound_ang': 100
        }

    def get(self, cls):
        return self._systems[cls]
        

    def reset(self, rng: np.random.Generator):
        mujoco.mj_resetData(self.model, self.data)

        for instance in self._systems.values():
            instance.reset(rng)

        #self.phys_context.kinematics.basis.update_rotation()

    def step(self, rng: np.random.Generator, action = None):
        if action is not None:
            self.data.ctrl[:] = action

        for _ in range(self.n_substeps): # Simulate control updates
            mujoco.mj_step(self.model, self.data)

        for instance in self._systems.values():
            instance.step(rng)

        #self.phys_context.kinematics.basis.update_rotation()

    def forward(self):
        mujoco.mj_forward(self.model, self.data)
    

