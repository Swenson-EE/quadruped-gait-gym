import gymnasium as gym
import numpy as np


class DeltaActionWrapper(gym.ActionWrapper):
    def __init__(self, env, max_delta):
        super().__init__(env)

        self.max_delta = max_delta

        self.prev_action = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.prev_action = None

        return obs, info
    
    def action(self, action):
        if self.prev_action is None:
            self.prev_action = action.copy()
            return action
        
        # 1. limit how much the action can change
        delta = action - self.prev_action
        delta = np.clip(delta, -self.max_delta, self.max_delta)

        # 2. Apply delta
        constrainted_action = self.prev_action + delta
        
        self.prev_action = constrainted_action.copy()
        return constrainted_action


