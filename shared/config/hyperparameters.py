
import torch

from dataclasses import dataclass, field

from shared.config.config_cls import ConfigCls


@dataclass
class Hyperparameters(ConfigCls):
    learning_rate: float = 1e-3
    batch_size: int = 64
    batch_steps: int = 2048
    discount_factor: float = 0.99
    entropy_coeff: float = 0.0
    activation: str = 'tanh'
