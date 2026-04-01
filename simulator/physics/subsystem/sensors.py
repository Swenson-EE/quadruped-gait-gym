import mujoco
import numpy as np

from simulator.physics.core import PhysicsSubsystem
from simulator.physics.subsystem.sensor_systems import SensorModel

class Sensors(PhysicsSubsystem):

    def __init__(self, context):
        super().__init__(context)

        

        self._gyro = SensorModel(
            self.context, 
            'gyro',
            self.info.sensor_ids[self.info.sensor_gyro],
            max_value=20.0
            #max_value=np.rad2deg(30.0)
        )

        self._accel = SensorModel(
            self.context, 
            'accel',
            self.info.sensor_ids[self.info.sensor_accel],
            max_value=200.0
            #max_value=(30.0 * 1000)
        )

        self._quat = SensorModel(
            self.context,
            'quat', 
            self.info.sensor_ids[self.info.sensor_quat],
            max_value=1.0
        )

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
        self.imu_gyro.read()
        self.imu_accel.read()
        self.imu_quat.read()

    def reset(self, rng):
        self._read()

    def step(self, rng):
        self._read()
