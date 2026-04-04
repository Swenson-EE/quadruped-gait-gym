from .sensor_model import SensorModel


class NoisySensorModel(SensorModel):

    def __init__(self, sim, name, sensor_id, max_value=1.0, std=0.1, bias_std=0.0):
        super().__init__(sim, name, sensor_id, max_value)

        self._std = std
        self._bias_std = bias_std

        self._bias = 0.0
        self._noise = 0.0

    def reset(self, rng):
        self._bias = rng.normal(0, self._bias_std, size=self._shape)

    def step(self, rng):
        self._noise = rng.normal(0, self._std, size=self._shape)

    def _get_value(self):
        true_value = super()._get_value()
        noisy_value = true_value + self._bias + self._noise
        
        return noisy_value
    



