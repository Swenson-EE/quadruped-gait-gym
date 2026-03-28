from loggers.base.periodic_logger import PeriodicLogger


class PeriodicAverageLogger(PeriodicLogger):
    def row_operation(self, name, values, length):
        return [(name, sum(values) / length)]

