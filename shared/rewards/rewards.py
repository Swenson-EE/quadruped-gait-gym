from dataclasses import dataclass


@dataclass
class Rewards:
    WT_Movement: float = 1000
    WT_LateralMovement: float = 1
    WT_Velocity: float = 100
    WT_Instability: float = 0.1
    WT_InstabilityRate: float = 0.001
    WT_Bouncing: float = 0.5
    WT_Slip: float = 0.1
    WT_Crawl: float = 0.1
    WT_Smooth: float = 1.0
    WT_Tilt: float = 1.0
    WT_Jitter: float = 1.0
    WT_Clearance: float = 0.1
    WT_Inactive: float = 0.1
    WT_Living: float = 0.1

    PAW_Z_THRESHOLD: float = 0.01
