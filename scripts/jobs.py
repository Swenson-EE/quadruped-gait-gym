from shared.algorithm.algorithm_types import Algorithm
from runner.training_job import TrainingJob

run = 'sequential'
parallel_run = 2 # For use in run='parallel'

parallel_env=8
steps=1e4
seed = 10
discount_factor = 0.99
learning_rate = 1e-3
rec_freq = 5
num_checkpoints = 1

network_architecture = [64]*4


training_jobs = {
    algorithm: [
        TrainingJob(
            algo=algorithm.value,
            total_steps=int(steps),
            parallel_env=parallel_env,
            seed=seed,

            recording_frequency=rec_freq,

            net_arch=network_architecture,

            lr=learning_rate,
            discount_factor=discount_factor
        ) for _ in range(num_checkpoints)
    ] for algorithm in Algorithm
}


jobs: list[TrainingJob] = training_jobs[Algorithm.DDPG] 