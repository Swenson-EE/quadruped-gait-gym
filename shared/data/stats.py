from enum import Enum


class Statistics(str, Enum):
    MEAN = 'mean'
    STD = 'std'
    MIN = 'min'
    MAX = 'max'
    