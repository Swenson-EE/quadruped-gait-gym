# Weights for rewards and penalties
WT_Movement = 10                          # Weight for forward movement reward
WT_Velocity = 1
WT_Instability = 0.1                         # Weight for instability penalty
WT_Bouncing = 0.1                         # Weight for bouncing penalty
WT_Slip = 0.1                            # Weight for slipping penalty
WT_Crawl = 0.1                           # Weight for crawling behavior penalty
WT_Smooth = 1.0                          # Weight for movement smoothness penalty
WT_Tilt = 1.0                           # Weight for tilt penalty
WT_Jitter = 1.0                           # Weight for movement jitter penalty
WT_Clearance = 0.1                          # Weight for paw clearance penalty
PAW_Z_THRESHOLD = 0.01               # Threshold for paw height clearance (in meters)