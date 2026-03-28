from runner.parallel_runner import ParallelRunner
from runner.process_factory import ProcessFactory
from runner.sequential_runner import SequentialRunner
from scripts.jobs import run, parallel_run, jobs


if __name__ == "__main__":
    factory = ProcessFactory(script='shared.training.train')

    if run == 'sequential':
        job_runner = SequentialRunner(factory)
    elif run == 'parallel':
        job_runner = ParallelRunner(factory, max_parallel=parallel_run)
    else:
        print("Invalid run type")
        exit()

    job_runner.run(jobs)
    
