from scheduler.base import SchedulerConfig
from scheduler.manager import QueueManager

from runner.training_job import TrainingJob, training_job_parser


if __name__ == "__main__":

    args = training_job_parser.parse_args()
    job = TrainingJob(**vars(args))

    config = SchedulerConfig.from_json_file("scheduler_config.json")

    manager = QueueManager(address=(config.domain, int(config.port)), authkey=config.auth_key.encode())
    manager.connect()

    queue = manager.get_queue()
    
    print('job:', job)
    queue.put(job)
    print("Job submitted")

