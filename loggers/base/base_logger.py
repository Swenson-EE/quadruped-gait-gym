import pandas as pd

from typing import TypeVar, Generic
from dataclasses import dataclass

@dataclass
class LoggerParameters:
    name: str

T = TypeVar("T", bound=LoggerParameters)

class BaseLogger(Generic[T]):
    def __init__(self, params: T):
        self.params = params

    def get_name(self):
        return self.params.name

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
