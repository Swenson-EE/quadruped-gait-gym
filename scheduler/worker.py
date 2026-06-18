from runner.training_job import TrainingJob
from scheduler.base import SchedulerConfig
from scheduler.jobs.optimization_job import Optimizations, OptimizeJob
from scheduler.manager import QueueManager, job_queue, shutdown_flag

#from shared.training.train import train
import importlib
import shared.training.train as training_module
from shared.training.training_status import TrainingStatus

from scripts.optimize import Optimizer, PrimaryWeightOptimizer, SecondaryWeightOptimizer, HyperparameterOptimizer, ModelParameterOptimizer

import gc
import torch


def print_stage(stage, job, n=30):
    print("=" * n)
    print(f"[{stage}]")
    print(job)
    print("=" * n)


training_function = training_module.train

def training_job(job):
    # Don't need currently for straightthrough runs
    # try:
    #     importlib.reload(training_module)
    #     global training_function
    #     training_function = training_module.train
    # except Exception as e:
    #     print('[ERROR] reloading training module')
    #     training_function = None

    # if training_function is None:
    #     print("[ERROR] No training function available")
    #     return
    
    print_stage("START", job)
    status, message = training_function(job)
    
    if status == TrainingStatus.SUCCESS:
        print_stage("DONE", job)
    else:
        print_stage(status.value.upper(), message)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def optimize_job(job: OptimizeJob):
    try:
        print_stage("START", job)
        print(f"[OPTIMIZER JOB]: {job.optimizer}")

        OptimizerClass: Optimizer = None
        match job.optimizer:
            case Optimizations.PRIMARY_REWARDS:
                OptimizerClass = PrimaryWeightOptimizer
            case Optimizations.SECONDARY_REWARDS:
                OptimizerClass = SecondaryWeightOptimizer
            case Optimizations.HYPERPARAMETER:
                OptimizerClass = HyperparameterOptimizer
            case Optimizations.MODEL_PARAMETER:
                OptimizerClass = ModelParameterOptimizer

        if OptimizerClass is None:
            print('[ERROR] invalid optimization:', job.optimizer.value)
            return
        

        optimizer = OptimizerClass(args=job.args)
        optimizer.run()
    except Exception as e:
        print('[ERROR]', e)
    else:
        print_stage("DONE", job)




JOB_CLASS_MAP = {
    "train": TrainingJob,
    "optimize": OptimizeJob
}


def worker_loop(queue):

    print("Worker started...")
    while not shutdown_flag.is_set():
        try:
            job = queue.get(timeout=1)
        except:
            continue    # no job, loop again
        
        type = job["type"]
        payload = job["payload"]


        match type:
            case "train":
                training_job(payload)
            case "optimize":
                optimize_job(payload)
        
        cleanup()
            
        print(f"[QUEUE] Items left in queue {queue.qsize()}")
        print("\n", "~" * 40, "\n")

        
        

    print("Shutdown requested, stopping server.")
    # Shutdown worker...

if __name__ == "__main__":
    config = SchedulerConfig.from_json_file("scheduler_config.json")

    manager = QueueManager(address=(config.domain, int(config.port)), authkey=config.auth_key.encode())
    server = manager.get_server()

    print("Starting queue server...")

    # Run worker in same process
    import threading
    t = threading.Thread(name=config.name, target=worker_loop, args=(job_queue, ), daemon=True)
    t.start()

    server.serve_forever()
    


