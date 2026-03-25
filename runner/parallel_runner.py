from runner.process_factory import ProcessFactory
from runner.training_job import TrainingJob


class ParallelRunner:
    def __init__(self, factory: ProcessFactory, max_parallel=2):
        self.factory = factory
        self.max_parallel = max_parallel

    def run(self, jobs: list[TrainingJob]):
        processes = []

        for job in jobs:
            while len(processes) >= self.max_parallel:
                processes = [p for p in processes if p.poll() is None]

            p = self.factory.run(job, blocking=False)
            processes.append(p)

        for p in processes:
            p.wait()
