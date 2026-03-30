from scheduler.base import SchedulerConfig
from scheduler.manager import QueueManager

from runner.training_job import TrainingJob, training_job_parser, parse_args_to_dataclass


if __name__ == "__main__":

    job = parse_args_to_dataclass(training_job_parser, TrainingJob)

    config = SchedulerConfig.from_json_file("scheduler_config.json")

    manager = QueueManager(address=(config.domain, int(config.port)), authkey=config.auth_key.encode())
    manager.connect()

    queue = manager.get_queue()
    
    print('job:', job)
    queue.put(job)
    print("Job submitted")

