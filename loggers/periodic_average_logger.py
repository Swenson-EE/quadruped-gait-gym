from loggers.periodic_logger import PeriodicLogger


class PeriodicAverageLogger(PeriodicLogger):
    def row_operation(self, values, length):
        return sum(values) / length

