import pandas as pd
from dataclasses import dataclass, field
from typing import TypeVar, Generic

from loggers.base.base_logger import BaseLogger, LoggerParameters


@dataclass
class FlatPeriodicLoggerParameters(LoggerParameters):
    log_frequency: int = 10
    items_to_track: list[str] = field(default_factory=list)


U = TypeVar("U", bound=FlatPeriodicLoggerParameters)


class FlatPeriodicLogger(BaseLogger[U], Generic[U]):
    def __init__(self, params: U):
        super().__init__(params=params)

        self.episode_counts = None
        self.is_recording = None
        self.current_episodes = None

        self.finished_episodes = {}

    def on_training_start(self, num_envs):
        self.episode_counts = [0] * num_envs
        self.is_recording = [False] * num_envs

        # Just store floats per key now
        self.current_episodes = [
            {key: 0.0 for key in self.params.items_to_track}
            for _ in range(num_envs)
        ]

    def on_step(self, env_id, action, info, done):
        if self.is_recording[env_id] and info is not None:
            for key in self.params.items_to_track:
                self.current_episodes[env_id][key] += info.get(key, 0.0)

    def on_episode_end(self, env_id, ep_num):
        if self.is_recording[env_id]:
            if ep_num not in self.finished_episodes:
                self.finished_episodes[ep_num] = []

            # Copy flat dict
            self.finished_episodes[ep_num].append(
                dict(self.current_episodes[env_id])
            )

            # Reset values
            for key in self.params.items_to_track:
                self.current_episodes[env_id][key] = 0.0

            self.is_recording[env_id] = False

        # Decide if next episode should be recorded
        if (ep_num + 1) % self.params.log_frequency == 0:
            self.is_recording[env_id] = True

    def on_training_end(self):
        rows = []

        for ep_num in sorted(self.finished_episodes.keys()):
            env_dicts = self.finished_episodes[ep_num]
            ep_row = {"episode": ep_num}

            for key in self.params.items_to_track:
                values = [d[key] for d in env_dicts]

                columns = self.row_operation(
                    name=key,
                    values=values,
                    length=len(values),
                )

                for (name, value) in columns:
                    ep_row[name] = value

            rows.append(ep_row)

        return pd.DataFrame(rows)

    def row_operation(self, name, values, length) -> list[tuple[str, float]]:
        return [(name, 0.0)]
    
    