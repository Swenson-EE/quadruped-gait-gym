from dataclasses import dataclass
import json


@dataclass
class Rewards:
    WT_Movement: float = 10
    WT_LateralMovement: float = 1
    WT_Velocity: float = 1
    WT_Instability: float = 0.1
    WT_InstabilityRate: float = 1.0
    WT_Bouncing: float = 0.1
    WT_Slip: float = 0.1
    WT_Crawl: float = 0.01
    WT_Smooth: float = 1.0
    WT_Tilt: float =    0.1
    WT_Jitter: float = 1.0
    WT_Clearance: float = 0.1
    WT_Inactive: float = 0.1
    WT_Living: float = 0.1

    PAW_Z_THRESHOLD: float = 0.01


    @classmethod
    def from_dict(cls, data: dict) -> "Rewards":
        return cls(**data)

    @classmethod
    def from_json_file(cls, filepath: str) -> "Rewards":
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)