from dataclasses import dataclass

from shared.config.config_cls import ConfigCls


@dataclass
class EnvironmentParameters(ConfigCls):
    joint_min: int = -120
    joint_max: int = 120
    joint_delta: int = 15
    phase_rate: float = 1.0
    length_joint_history: int = 50
    episode_length: int = 250

    total_length: int = 1e6

    