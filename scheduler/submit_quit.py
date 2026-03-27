from scheduler.base import SchedulerConfig
from scheduler.manager import QueueManager

if __name__ == "__main__":
    config = SchedulerConfig.from_json_file("scheduler_config.json")

    manager = QueueManager(address=(config.domain, int(config.port)), authkey=config.auth_key.encode())
    manager.connect()

    print(manager.quit())