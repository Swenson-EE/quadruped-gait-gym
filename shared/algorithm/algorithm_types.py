from enum import Enum

class Algorithm(str, Enum):
    #QLEARNING = "qlearning"     # Q-Learning
    
    PPO_C = "ppo_c"             # Proximal Policy Optimization (Continuous)
    SAC = "sac"                 # Soft Actor-Critic
    DDPG = "ddpg"               # Deep Deterministic Policy Gradient

    A2C = "a2c"                 # Advantage Actor-Critic
    PPO_D = "ppo_d"             # Proximal Policy Optimization (Discrete)