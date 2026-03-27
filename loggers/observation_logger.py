from loggers.base.periodic_logger import PeriodicLogger


class ObservationLogger(PeriodicLogger):
    def __init__(self, log_frequency=10):
        super().__init__("Observations", log_frequency=log_frequency, items_to_track={
            'observation': ['gyro', 'accel']
        })

