from simulator.core.subsystem.modular_subsystem import ModularSubsystem
from simulator.core.registry import SubsystemRegistry

import mujoco
from types import SimpleNamespace

class RobotInfo:

    ground_plane_name: str = 'ground'

    joint_names: list[str] = [
        "front_left_hip_joint", "front_left_knee_joint",
        "front_right_hip_joint", "front_right_knee_joint",
        "back_left_hip_joint", "back_left_knee_joint",
        "back_right_hip_joint", "back_right_knee_joint"
    ]

    arm_names: list[str] = [
        "front_left_lower", "front_right_lower", "back_left_lower", "back_right_lower"
    ]

    feet_names: list[str] = [
        "front_left_foot", "front_right_foot", "back_left_foot", "back_right_foot"
    ]

    feet_site_names: list[str] = [
        "front_left_foot_site", "front_right_foot_site", "back_left_foot_site", "back_right_foot_site"
    ]

    sensor_quat: str = "imu_quat"
    sensor_gyro: str = "imu_gyro"
    sensor_accel: str = "imu_accel"

    sensor_ids = None


    def __init__(self, model):
        self.model = model


        self.ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.ground_plane_name)

        self.foot_geom_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in self.feet_names]

        self.foot_site_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name) for name in self.feet_site_names]

        self.foot_radii = {
            geom_id: self.model.geom_size[geom_id][0] for geom_id in self.foot_geom_ids
        }

        self.arm_geom_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in self.arm_names]

        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]

        #self.joint_qpos_ids = [self.model.jnt_qposadr[joint_id] for joint_id in self.joint_ids]
        
        self.joint = SimpleNamespace(
            ids = joint_ids,
            qpos_addr = [self.model.jnt_qposadr[joint_id] for joint_id in joint_ids],
            qvel_addr = [self.model.jnt_dofadr[joint_id] for joint_id in joint_ids]
        )
        


        self.sensor_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name) for name in [self.sensor_quat, self.sensor_gyro, self.sensor_accel] 
        }
    
        