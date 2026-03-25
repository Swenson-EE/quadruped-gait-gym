from runner.parallel_runner import ParallelRunner
from runner.process_factory import ProcessFactory
from runner.sequential_runner import SequentialRunner

from jobs import run, parallel_run, jobs


if __name__ == "__main__":
    factory = ProcessFactory(script='train.py')

    if run == 'sequential':
        runner = SequentialRunner(factory)
    elif run == 'parallel':
        runner = ParallelRunner(factory, max_parallel=parallel_run)
    else:
        print("Invalid run type")
        exit()

    runner.run(jobs)
    
