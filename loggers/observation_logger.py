from loggers.base.periodic_logger import PeriodicLogger, PeriodicLoggerParameters


class ObservationLogger(PeriodicLogger):
    def __init__(self, log_frequency=10):
        super().__init__(params=PeriodicLoggerParameters(
            name="Observations",
            log_frequency=log_frequency,
            items_to_track={
                'observation': ['gyro', 'accel']
            }
        ))

