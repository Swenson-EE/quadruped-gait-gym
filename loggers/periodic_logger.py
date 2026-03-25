import pandas as pd

from loggers.base_logger import BaseLogger


class PeriodicLogger(BaseLogger):
    def __init__(self, name, log_frequency=10, items_to_track = {}):
        super().__init__(name)

        self.log_frequency = log_frequency

        self.episode_counts = None
        self.is_recording = None
        self.current_episodes = None

        self.items_to_track = items_to_track

        self.finished_episodes = {}

    def on_training_start(self, num_envs):
        self.episode_counts = [0] * num_envs
        self.is_recording = [False] * num_envs

        self.current_episodes = []
        for _ in range(num_envs):
            environment_dict = {}
            for main_key, sub_keys in self.items_to_track.items():
                environment_dict[main_key] = {k: 0 for k in sub_keys}
            self.current_episodes.append(environment_dict)

    def on_step(self, env_id, action, info, done):
        if self.is_recording[env_id] and info is not None:
                for main_key, sub_keys in self.items_to_track.items():
                    sub_dict = info.get(main_key, {})
                    for sub_key in sub_keys:
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
            for main_key, sub_keys in self.items_to_track.items():
                for sub_key in sub_keys:
                    self.current_episodes[env_id][main_key][sub_key] = 0
            self.is_recording[env_id] = False
        
        # Decide if next episode should be recorded
        if (ep_num + 1) % self.log_frequency == 0:
            self.is_recording[env_id] = True

    def on_training_end(self):
        # Average across envs per episode
        averaged_rows = []
        for ep_num in sorted(self.finished_episodes.keys()):
            env_dicts = self.finished_episodes[ep_num]
            avg_row = {"episode": ep_num}

            for main_key, sub_keys in self.items_to_track.items():
                for sub_key in sub_keys:
                    # Average across envs
                    avg_row[f"{main_key}_{sub_key}"] = sum(
                        d[main_key][sub_key] for d in env_dicts
                    ) / len(env_dicts)

            averaged_rows.append(avg_row)
        
        return pd.DataFrame(averaged_rows)
