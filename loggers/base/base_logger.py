import pandas as pd


class BaseLogger:
    def __init__(self, name: str):
        self.name = name

    def on_training_start(self, num_envs: int):
        """Called when training starts"""
        pass
    
    def on_step(self, env_id, action, info, done):
        """Called every step"""
        pass

    def on_episode_end(self, env_id, ep_num):
        """Called when an episode ends"""
        pass
    
    def on_training_end(self):
        """Returns a pandas Dataframe"""
        return pd.DataFrame()
