import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from loggers.base_logger import BaseLogger

class MultiLoggerCallback(BaseCallback):
    def __init__(self, name, algo, loggers: list[BaseLogger], verbose=0):
        super().__init__(verbose)
        self.name = name
        self.algo = algo
        self.loggers = loggers

        self.episode_counts = None
        

    def _on_training_start(self):
        self.num_envs = self.training_env.num_envs
        self.episode_counts = [0] * self.num_envs
        
        for logger in self.loggers:
            logger.on_training_start(self.num_envs)
        

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        actions = self.locals["actions"]

        for i in range(self.num_envs):
            for logger in self.loggers:
                logger.on_step(i, actions[i], infos[i], dones[i])

            if dones[i]:
                for logger in self.loggers:
                    logger.on_episode_end(env_id=i, ep_num=self.episode_counts[i])
                    
                self.episode_counts[i] += 1

        return True
    

    def _on_training_end(self):
        if len(self.loggers) > 0:
            import os
            save_dir = os.path.join("saved", self.algo, "data")

            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            excel_path = f'{save_dir}/{self.name}_rewards.xlsx'

            with pd.ExcelWriter(excel_path) as writer:
                for logger in self.loggers:
                    df = logger.on_training_end()
                    df.to_excel(writer, sheet_name=logger.name, index=False)
