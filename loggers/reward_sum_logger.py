from loggers.base.flat_periodic_logger import FlatPeriodicLogger, FlatPeriodicLoggerParameters



class RewardSumLogger(FlatPeriodicLogger):
    def __init__(self, log_frequency=10, running_average = 10):
        super().__init__(
            params=FlatPeriodicLoggerParameters(
                name="Reward Sum",
                log_frequency=1,
                items_to_track=["reward_sum", "penalty_sum"]
            )
        )

    def row_operation(self, name, values, length):
        print('='*10)
        print('name:', name)
        print('values:', values)
        print('length:', length)
        print('='*10)

        return [(name, sum(values) / length)]

