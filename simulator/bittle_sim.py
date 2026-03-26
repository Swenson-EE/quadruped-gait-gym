from dataclasses import dataclass, field

import mujoco
import numpy as np


@dataclass
class BittleParameters:
    model_path: str = ''

    ground_plane_name: str = 'ground'

    joint_names: list[str] = field(default_factory=lambda: [
        "front_left_hip_joint", "front_left_knee_joint",
        "front_right_hip_joint", "front_right_knee_joint",
        "back_left_hip_joint", "back_left_knee_joint",
        "back_right_hip_joint", "back_right_knee_joint"
    ])

    arm_names: list[str] = field(default_factory=lambda: [
        "front_left_lower", "front_right_lower", "back_left_lower", "back_right_lower"
    ])

    feet_names: list[str] = field(default_factory=lambda: [
        "front_left_foot", "front_right_foot", "back_left_foot", "back_right_foot"
    ])

    sensor_quat: str = "imu_quat"
    sensor_gyro: str = "imu_gyro"
    sensor_accel: str = "imu_accel"

    fall_angle = 1.3

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

        self.ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, parameters.ground_plane_name)

        self.foot_geom_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in parameters.feet_names]

        self.foot_radii = {
            geom_id: self.model.geom_size[geom_id][0] for geom_id in self.foot_geom_ids
        }

        self.arm_geom_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in parameters.arm_names]

        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in parameters.joint_names]

        self.sensor_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name) for name in [parameters.sensor_quat, parameters.sensor_gyro, parameters.sensor_accel] 
        }

        self.n_substeps = int(parameters.control_dt / self.model.opt.timestep)      # The number of substeps to take in a single physics step to simulate control delay

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self._update_rotation_matrix()

    def step(self, action):
        self.data.ctrl[:] = action

        for _ in range(self.n_substeps): # Simulate control updates
            mujoco.mj_step(self.model, self.data)

        self._update_rotation_matrix()

    def get_sensor(self, name):        
        sensor_id = self.sensor_ids[name]
        
        start = self.model.sensor_adr[sensor_id]
        end = start + self.model.sensor_dim[sensor_id]

        return self.data.sensordata[start:end].copy()
            
    def get_joint_angles(self):
        return np.rad2deg([
            self.data.qpos[joint_id] for joint_id in self.joint_ids
        ]).astype(int)

    def get_position(self):
        return self.data.qpos[0:3].copy() # x, y, z position (m)

    def get_velocity(self):
        return self.data.qvel[0:3].copy() # Linear velocity in world frame (m/s)
    
    def _update_rotation_matrix(self):
        quat = self.data.qpos[3:7].copy() # Quaternion (w, x, y, z)
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, quat) # Rotation matrix from world frame to local frame
        self._rot_mat = rot_mat.reshape(3, 3)

    def get_rotation_matrix(self):
        return self._rot_mat

    def world_to_local(self, value):
        return self._rot_mat @ value

    def get_tilt(self):
        w, x, y, z = self.get_sensor(self.params.sensor_quat)
        
        roll = np.arctan2(
            2*(w*x + y*z),
            1 - 2*(x**2 + y**2)
        )

        pitch = np.arctan2(
            2*(w*y - z*x),
            np.sqrt(1 - (2*(w*y - z*x))**2)
        )

        return roll, pitch
    
    def get_tilt_rate(self):
        imu_gyro = self.get_sensor(self.params.sensor_gyro)
        roll_rate, pitch_rate, _ = imu_gyro
        return roll_rate, pitch_rate
    
    
    def get_feet_z(self):
        feet_z = { geom_id: self.data.geom_xpos[geom_id][2] for geom_id in self.foot_geom_ids }
        return feet_z
    
    def get_num_contacting_arms(self):
        num_arms_contacting = 0
        for c in self.data.contact[:self.data.ncon]:
            g1, g2 = c.geom1, c.geom2 # The touching geometry IDs

            if (self.ground_id in (g1, g2)) and (g1 in self.arm_geom_ids or g2 in self.arm_geom_ids):
                num_arms_contacting += 1
        
        return num_arms_contacting

    
    def is_fallen(self):
        fall_angle = self.params.fall_angle
        roll, pitch = self.get_tilt()

        is_fallen = (np.fabs(roll) > fall_angle) or (np.fabs(pitch) > fall_angle)
        return is_fallen
    
    def is_moving(self, moving_threshold = 1e-3):
        return np.linalg.norm(self.get_velocity()) > moving_threshold
    
    

