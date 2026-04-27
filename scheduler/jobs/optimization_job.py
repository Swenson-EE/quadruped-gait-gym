from dataclasses import dataclass, field
from enum import Enum

from scripts.optimize.optimize import OptimizeArguments


@dataclass
class Optimizations(str, Enum):
    PRIMARY_REWARDS = "primary-rewards"
    SECONDARY_REWARDS = "secondary-rewards"

    HYPERPARAMETER = "hyperparameters"
    MODEL_PARAMETER = "model-parameters"


@dataclass
class OptimizeJob:
    optimizer: Optimizations = Optimizations.PRIMARY_REWARDS

    args: OptimizeArguments = field(default_factory=OptimizeArguments)

    