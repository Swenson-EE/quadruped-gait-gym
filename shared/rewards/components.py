from dataclasses import dataclass
import numpy as np


@dataclass
class RewardComponents:
    position: np.ndarray
    velocity: np.ndarray
    
    jitter_1st_order: np.ndarray
    jitter_2nd_order: np.ndarray

    imu_gyro: np.ndarray
    imu_accel: np.ndarray

    paw_clearance: dict[int, np.float64]
    num_arms_contacting: int
    
    roll: np.float64
    pitch: np.float64
