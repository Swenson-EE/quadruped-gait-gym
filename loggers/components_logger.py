from loggers.base.statistics_logger import StatisticsLogger, StatisticsLoggerParameters
from shared.data.stats import Statistics


class ComponentsLogger(StatisticsLogger):
    def __init__(self, log_frequency=10):
        super().__init__(params=StatisticsLoggerParameters(
            name="Components",
            log_frequency=log_frequency,
            items_to_track={
                "components": [
                    "position", "velocity",
                    "jitter_1st_order", "jitter_2nd_order",
                    "paw_clearance",
                    "roll", "pitch"
                ]
            },
            stats_to_track=[
                Statistics.MEAN,
                Statistics.STD,
                Statistics.MIN,
                Statistics.MAX
            ]
        ))