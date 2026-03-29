from loggers.base.statistics_logger import StatisticsLogger, StatisticsLoggerParameters, Statistics


class ObservationLogger(StatisticsLogger):
    def __init__(self, log_frequency=10):
        super().__init__(params=StatisticsLoggerParameters(
            name="Observations",
            log_frequency=log_frequency,
            items_to_track=['observation'],
            stats_to_track=[
                Statistics.MEAN,
                Statistics.STD,
                Statistics.MIN,
                Statistics.MAX
            ]
        ))

