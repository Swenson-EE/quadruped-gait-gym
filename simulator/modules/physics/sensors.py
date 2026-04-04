import mujoco
import numpy as np

from simulator.core.subsystem.subsystem import Subsystem
from simulator.core.registry import register_module
from simulator.modules.physics_module import Physics
from simulator.core.robot_info import RobotInfo

from simulator.modules.physics.sensor_systems import SensorModel, NoisySensorModel


@register_module(Physics)
class Sensors(Subsystem):

    _sensors = {}
        
    def initialize(self):
        info = self.sim.robot_info

        self.create_sensor(
            NoisySensorModel, 
            'gyro', 
            info.sensor_ids[info.sensor_gyro],
            max_value=20.0, 
            #max_value=np.rad2deg(30.0),
            std=0.005, 
            bias_std=0.01
        )

        self.create_sensor(
            NoisySensorModel,
            'accel',
            info.sensor_ids[info.sensor_accel],
            max_value=200.0,
            #max_value=(30.0 * 1000)
            std=0.05,
            bias_std=0.1
        )

        self.create_sensor(
            SensorModel,
            'quat', 
            info.sensor_ids[info.sensor_quat],
            max_value=1.0
        )


    def create_sensor(self, cls, name, sensor_id, **kwargs):
        sensor = cls(self.sim, name, sensor_id, **kwargs)
        self._sensors[name] = sensor
        return sensor
    

    @property
    def imu_gyro(self):
        return self._sensors['gyro']
    
    @property
    def imu_accel(self):
        return self._sensors['accel']
    
    @property
    def imu_quat(self):
        return self._sensors['quat']
    

    def _read(self):
        for sensor in self._sensors.values():
            sensor.read()

    def reset(self, rng):
        for sensor in self._sensors.values():
            sensor.reset(rng)

        self._read()

    def step(self, rng):
        for sensor in self._sensors.values():
            sensor.step(rng)
        self._read()
