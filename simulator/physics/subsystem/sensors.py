import mujoco
import numpy as np

from simulator.physics.core import PhysicsSubsystem
from simulator.physics.subsystem.sensor_systems import SensorModel, NoisySensorModel

class Sensors(PhysicsSubsystem):

    def __init__(self, context):
        super().__init__(context)

        

        self._gyro = NoisySensorModel(
            self.context, 
            'gyro',
            self.info.sensor_ids[self.info.sensor_gyro],
            max_value=20.0,
            std=0.005,
            bias_std=0.01
            #max_value=np.rad2deg(30.0)
        )

        self._accel = NoisySensorModel(
            self.context, 
            'accel',
            self.info.sensor_ids[self.info.sensor_accel],
            max_value=200.0,
            std=0.05,
            bias_std=0.1
            #max_value=(30.0 * 1000)
        )

        self._quat = SensorModel(
            self.context,
            'quat', 
            self.info.sensor_ids[self.info.sensor_quat],
            max_value=1.0
        )

        self._sensors: list[SensorModel] = [self._gyro, self._accel, self._quat]


    @property
    def imu_gyro(self):
        return self._gyro
    
    @property
    def imu_accel(self):
        return self._accel
    
    @property
    def imu_quat(self):
        return self._quat

    def _read(self):
        for sensor in self._sensors:
            sensor.read()

    def reset(self, rng):
        for sensor in self._sensors:
            sensor.reset(rng)

        self._read()

    def step(self, rng):
        for sensor in self._sensors:
            sensor.step(rng)
        self._read()
