from loggers.periodic_logger import PeriodicLogger


class RewardLogger(PeriodicLogger):
    def __init__(self, log_frequency=10):
        super().__init__("Rewards", log_frequency=log_frequency, items_to_track={
            'reward': ['movement', 'velocity'],
            'penalty': ['movement', 'velocity', 'smooth', 'clearance', 'crawling', 'stability_angle', 'stability_rate', 'z_bounce']
        })

