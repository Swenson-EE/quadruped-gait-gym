import mujoco

from simulator.physics.core import PhysicsSubsystem


class Sensors(PhysicsSubsystem):
    _sensor_slices = {}

        
    def _get_sensor_slice(self, sensor_id):
        if sensor_id not in self._sensor_slices:
            self._sensor_slices[sensor_id] = slice(
                self.model.sensor_adr[sensor_id],
                self.model.sensor_adr[sensor_id] + self.model.sensor_dim[sensor_id]
            )

        return self._sensor_slices[sensor_id]

    def get_sensor(self, sensor_id):
        s = self._get_sensor_slice(sensor_id)
        return self.data.sensordata[s]
    

    @property
    def imu_gyro(self):
        gyro_id = self.info.sensor_ids[self.info.sensor_gyro]
        return self.get_sensor(gyro_id)
    
    @property
    def imu_accel(self):
        accel_id = self.info.sensor_ids[self.info.sensor_accel]
        return self.get_sensor(accel_id)
    
    @property
    def imu_quat(self):
        quat_id = self.info.sensor_ids[self.info.sensor_quat]
        return self.get_sensor(quat_id)
