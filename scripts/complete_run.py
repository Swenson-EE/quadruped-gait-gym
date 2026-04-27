
from runner.training_job import TrainingJob
from scheduler.jobs.optimization_job import Optimizations, OptimizeJob, OptimizeArguments
from scheduler.submit_job import submit_to_queue
from shared.algorithm.algorithm_types import Algorithm



def submit_algorithm_jobs():
    model_jobs = []
    net_archs = [[64, 64], [64, 64, 64, 64], [128, 128, 128, 128], [256, 256, 256, 256]]


    for algo in Algorithm:
        for net_arch in net_archs:
            model_jobs.append(TrainingJob(
                algo=algo,
                net_arch=net_arch,
                parallel_env=8,
                total_steps=1e6
            ))

    for job in model_jobs:
        submit_to_queue('train', job)

def submit_optimize_jobs():
    optimize_jobs = [
        OptimizeJob(
            optimizer=optimization,
            args = OptimizeArguments(
                n_steps=10_000
            )
        ) 
        for optimization in Optimizations
    ]

    for job in optimize_jobs:
        submit_to_queue('optimize', job)


if __name__ == "__main__":
    # submit_algorithm_jobs()
    submit_optimize_jobs()

    


    
    




