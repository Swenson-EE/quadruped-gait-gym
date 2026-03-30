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

    paw_clearance: np.ndarray
    num_arms_contacting: int
    num_paws_contacting: int
    paw_slipping: np.ndarray
    
    roll: np.float64
    pitch: np.float64

    joint_variance: int


@dataclass
class RewardNormalizationFactors:
    position: float
    velocity: float

    jitter_1st_order: float
    jitter_2nd_order: float

    imu_gyro: float
    imu_accel: float

    paw_clearance: float
    num_arms_contacting: float
    paw_slipping: float

    roll: float
    pitch: float




def normalize_reward_components(rc: RewardComponents, factors: RewardNormalizationFactors) -> RewardComponents:
    normalized_fields = {}
    
    for field in rc.__dataclass_fields__:
        value = getattr(rc, field)
        factor = getattr(factors, field)
        
        if isinstance(value, (np.ndarray, list)):
            normalized_fields[field] = np.array(value, dtype=np.float64) / factor
        else:
            normalized_fields[field] = float(value) / factor
    
    return RewardComponents(**normalized_fields)


