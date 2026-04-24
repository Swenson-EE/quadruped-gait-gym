from dataclasses import dataclass, field, fields
import json
from dacite import from_dict
from typing import TypeVar, Type

from shared.utils import build_from_flat

# @dataclass
# class Rewards:
#     WT_Movement: float = 10
#     WT_LateralMovement: float = 1
#     WT_Velocity: float = 1
#     WT_Instability: float = 0.1
#     WT_InstabilityRate: float = 1.0
#     WT_Bouncing: float = 0.1
#     WT_Slip: float = 0.1
#     WT_Crawl: float = 0.01
#     WT_Smooth: float = 1.0
#     WT_Tilt: float =    0.1
#     WT_Jitter: float = 1.0
#     WT_Clearance: float = 0.1
#     WT_Inactive: float = 0.1
#     WT_Living: float = 0.1

#     PAW_Z_THRESHOLD: float = 0.01


#     @classmethod
#     def from_dict(cls, data: dict) -> "Rewards":
#         return cls(**data)

#     @classmethod
#     def from_json_file(cls, filepath: str) -> "Rewards":
#         with open(filepath, "r") as f:
#             data = json.load(f)
#         return cls.from_dict(data)



    
    


@dataclass
class PrimaryRewards:
    efficiency: float = 0.0
    air_time: float = 0.0
    forward_velocity: float = 0.0



    


@dataclass
class PrimaryPenalties:
    action_smooth: float = 0.0
    height: float = 0.0
    symmetry: float = 0.0

    stability_angle: float = 0.0
    stability_rate: float = 0.0


@dataclass 
class SecondaryRewards:
    bent_joint: float = 0.0


@dataclass
class SecondaryPenalties:
    crawling: float = 0.0

    joint_velocity: float = 0.0
    joint_acceleration: float = 0.0
    joint_jerk: float = 0.0
    joint_energy: float = 0.0

    large_joint_variance: float = 0.0

    lateral_movement: float = 0.0

    jitter_1st: float = 0.0
    jitter_2nd: float = 0.0

    lateral_velocity: float = 0.0
    z_velocity: float = 0.0


@dataclass
class Reward:
    primary: PrimaryRewards = field(default_factory=PrimaryRewards)
    primary_scale: float = 0.0

    secondary: SecondaryRewards = field(default_factory=SecondaryRewards)
    secondary_scale: float = 0.0


@dataclass
class Penalty:
    primary: PrimaryPenalties = field(default_factory=PrimaryPenalties)
    primary_scale: float = 0.0
    
    secondary: SecondaryPenalties = field(default_factory=SecondaryPenalties)
    secondary_scale: float = 0.0


@dataclass
class RewardWeights:

    reward: Reward = field(default_factory=Reward)
    penalty: Penalty = field(default_factory=Penalty)


    @classmethod
    def from_json_file(cls, filepath: str) -> "RewardWeights":
        with open(filepath, "r") as f:
            data = json.load(f)

        return from_dict(data_class=RewardWeights, data=data)

    @classmethod
    def from_dict(cls, data: dict):
        return from_dict(data_class=cls, data=data)

    @classmethod
    def from_flat_dict(cls, flat: dict) -> "RewardWeights":
        with open("config/reward_weights.json") as f:
            defaults = json.load(f)

        return build_from_flat(cls, flat, defaults=defaults)
    

def sum_group(weights: RewardWeights, values: dict):
    total = 0.0
    for f in fields(weights):
        weight = getattr(weights, f.name)
        value = values.get(f.name, 0.0)
        total += weight * value

    return total

def compute_total(weights: RewardWeights, rewards: dict, penalties: dict):
    # --- Rewards ---
    primary_reward = sum_group(weights.reward.primary, rewards)
    secondary_reward = sum_group(weights.reward.secondary, rewards)

    reward_total = (
        weights.reward.primary_scale * primary_reward +
        weights.reward.secondary_scale * secondary_reward
    )

    # --- Penalties ---
    primary_penalty = sum_group(weights.penalty.primary, penalties)
    secondary_penalty = sum_group(weights.penalty.secondary, penalties)

    penalty_total = (
        weights.penalty.primary_scale * primary_penalty +
        weights.penalty.secondary_scale * secondary_penalty
    )

    # --- Final ---
    total = reward_total - penalty_total

    return total, reward_total, penalty_total
