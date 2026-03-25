from runner.process_factory import ProcessFactory
from runner.training_job import TrainingJob


class SequentialRunner:
    def __init__(self, factory: ProcessFactory):
        self.factory = factory

    def run(self, jobs: list[TrainingJob]):
        for job in jobs:
            self.factory.run(job, blocking=True)
