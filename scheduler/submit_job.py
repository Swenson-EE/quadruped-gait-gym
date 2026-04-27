import argparse
from dataclasses import asdict, dataclass

from scheduler.base import SchedulerConfig
from scheduler.jobs.optimization_job import OptimizeJob
from scheduler.manager import QueueManager

from runner.training_job import TrainingJob
from shared.utils.dataclass_parser import build_parser_from_dataclass, parse_args_to_dataclass

import json


with open("config/training_job_defaults.json") as file:
    job_defaults = json.load(file)






parser = argparse.ArgumentParser(description="Submit Jobs")
subparsers = parser.add_subparsers(dest="job_type", required=True)

# ---- Train ----
train_parser = subparsers.add_parser("train")
training_job_parser = build_parser_from_dataclass(TrainingJob, parser=train_parser, defaults=job_defaults)

# ---- Optimize ----
optimize_parser = subparsers.add_parser("optimize")
build_parser_from_dataclass(OptimizeJob, parser=optimize_parser)


JOB_CLASS_MAP = {
    "train": TrainingJob,
    "optimize": OptimizeJob
}

PARSER_MAP = {
    "train": training_job_parser,
    "optimize": optimize_parser
}



def submit_to_queue(job_type: str, job):
    config = SchedulerConfig.from_json_file("scheduler_config.json")

    manager = QueueManager(
        address=(config.domain, int(config.port)),
        authkey=config.auth_key.encode()
    )
    manager.connect()

    queue = manager.get_queue()

    queue.put({
        "type": job_type,
        "payload": job
    })

    print(f"[SUBMITTED] {job_type}: {job}")


if __name__ == "__main__":
    args = parser.parse_args()

    job_type = args.job_type
    job_class = JOB_CLASS_MAP[job_type]

    job = parse_args_to_dataclass(args, job_class)
    
    submit_to_queue(job_type, job)


