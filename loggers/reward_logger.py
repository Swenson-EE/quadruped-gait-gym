#from loggers.periodic_logger import PeriodicLogger
from loggers.base.periodic_average_logger import PeriodicAverageLogger
from loggers.base.periodic_logger import PeriodicLoggerParameters

class RewardLogger(PeriodicAverageLogger):
    def __init__(self, log_frequency=10):
        super().__init__(params=PeriodicLoggerParameters(
            name="Rewards",
            log_frequency=log_frequency,
            items_to_track=['reward', 'penalty']
        ))

