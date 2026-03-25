from runner.training_job import TrainingJob


run = 'sequential'
parallel_run = 2 # For use in run='parallel'

parallel_env=16
steps = 1e6

jobs: list[TrainingJob] = [
    TrainingJob(algo="ppo_c", total_steps=int(steps), parallel_env=parallel_env) for _ in range(5)
]
