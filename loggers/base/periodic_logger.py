import pandas as pd
from dataclasses import dataclass, field
from typing import TypeVar, Generic

from loggers.base.base_logger import BaseLogger, LoggerParameters


@dataclass
class PeriodicLoggerParameters(LoggerParameters):
    log_frequency: int = 10
    items_to_track: list = field(default_factory=list)

U = TypeVar("U", bound=PeriodicLoggerParameters)

class PeriodicLogger(BaseLogger[U], Generic[U]):
    def __init__(self, params: U):
        super().__init__(params=params)

        self.episode_counts = None
        self.is_recording = None
        self.current_episodes = None

        self.finished_episodes = {}

    def on_training_start(self, num_envs):
        self.episode_counts = [0] * num_envs
        self.is_recording = [False] * num_envs

        self.current_episodes = []
        for _ in range(num_envs):
            environment_dict = {}
            for main_key in self.params.items_to_track:
                environment_dict[main_key] = {}
            self.current_episodes.append(environment_dict)

    def on_step(self, env_id, action, info, done):
        if self.is_recording[env_id] and info is not None:
                for main_key in self.params.items_to_track:
                    sub_dict = info.get(main_key, {})
                    for sub_key in sub_dict.keys():
                        if sub_key not in self.current_episodes[env_id][main_key]:
                            self.current_episodes[env_id][main_key][sub_key] = 0


                        self.current_episodes[env_id][main_key][sub_key] += sub_dict.get(sub_key, 0)
                    

    def on_episode_end(self, env_id, ep_num):

        if self.is_recording[env_id]:
            # Save this env's episode for averaging later
            if ep_num not in self.finished_episodes:
                self.finished_episodes[ep_num] = []
            
            # Deep copy of nested dict
            self.finished_episodes[ep_num].append(
                {k: dict(v) for k, v in self.current_episodes[env_id].items()}
            )

            # Reset
            for main_key in self.params.items_to_track:
                sub_dict = self.current_episodes[env_id].get(main_key, {})
                for sub_key in sub_dict.keys():
                    self.current_episodes[env_id][main_key][sub_key] = 0

            self.is_recording[env_id] = False
        
        # Decide if next episode should be recorded
        if (ep_num + 1) % self.params.log_frequency == 0:
            self.is_recording[env_id] = True

    def on_training_end(self):
        rows = []
        for ep_num in sorted(self.finished_episodes.keys()):
            env_dicts = self.finished_episodes[ep_num]
            ep_row = {"episode": ep_num}

            for main_key in self.params.items_to_track:
                sub_keys = set()
                for d in env_dicts:
                    sub_keys.update(d.get(main_key, {}.keys()))

                for sub_key in sub_keys:
                    columns = self.row_operation(
                        name=f"{main_key}_{sub_key}", 
                        values=[d[main_key][sub_key] for d in env_dicts], 
                        length=len(env_dicts)
                    )
                    
                    for (name, value) in columns:
                        ep_row[name] = value

            rows.append(ep_row)

        return pd.DataFrame(rows)
    
    def row_operation(self, name, values, length) -> list[tuple[str, float]]:
        return []


