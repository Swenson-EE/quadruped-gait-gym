#from simulator.physics.core.subsystem import PhysicsSubsystem
from simulator.core.subsystem import Subsystem
from simulator.core.transforms import TransformableValue

import math
import numpy as np


class SensorModel(Subsystem, TransformableValue):
    _sensor_slice: slice = None
    
    def __init__(self, sim, name, sensor_id, max_value=1, shape=(3,)):
        Subsystem.__init__(self, sim)
        TransformableValue.__init__(self, shape=shape, scale=max_value)

        self._name = name
        self._sensor_id = sensor_id
        self._max_value = max_value

    
    def _get_sensor_slice(self):
        if self._sensor_slice is None:
            sensor_start = self.model.sensor_adr[self._sensor_id]
            sensor_dim = self.model.sensor_dim[self._sensor_id]
            self._sensor_slice = slice(
                sensor_start,
                sensor_start + sensor_dim
            )

        return self._sensor_slice

    def _get_value(self):
        return self.data.sensordata[self._get_sensor_slice()]

    def read(self):
        sensor_value = self._get_value()
        self.real.set(sensor_value)

        