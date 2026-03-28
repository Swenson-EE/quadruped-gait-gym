from enum import Enum
from dataclasses import dataclass, field

import numpy as np

from loggers.base.periodic_logger import PeriodicLogger, PeriodicLoggerParameters




class Statistics(str, Enum):
    MEAN = 'mean'
    STD = 'std'
    MIN = 'min'
    MAX = 'max'
    

@dataclass
class StatisticsLoggerParameters(PeriodicLoggerParameters):
    stats_to_track: list[Statistics] = field(default_factory=list)



class StatisticsLogger(PeriodicLogger[StatisticsLoggerParameters]):
    def __init__(self, params: StatisticsLoggerParameters):
        super().__init__(params=params)
        
    def row_operation(self, name, values, length):
        results = []
        for stat in self.params.stats_to_track:
            match stat:
                case Statistics.MEAN:
                    value = np.mean(values, axis=0)
                case Statistics.STD:
                    value = np.std(values, axis=0)
                case Statistics.MIN:
                    value = np.min(values, axis=0)
                case Statistics.MAX:
                    value = np.max(values, axis=0)
                case _:
                    raise ValueError(f"Invalid Statistics {stat.value}")
            results.append((f"{name}_{stat.value}", value))

        return results
