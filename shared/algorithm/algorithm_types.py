from enum import Enum

class Algorithm(str, Enum):
    QLEARNING = "qlearning"     # Q-Learning
    PPO_D = "ppo_d"             # Proximal Policy Optimization (Continuous)
    PPO_C = "ppo_c"             # Proximal Policy Optimization (Discrete)
    SAC = "sac"                 # Soft Actor-Critic
    A2C = "a2c"                 # Advantage Actor-Critic
    DDPG = "ddpg"               # Deep Deterministic Policy Gradient