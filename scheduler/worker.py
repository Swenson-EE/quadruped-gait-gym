from scheduler.base import SchedulerConfig
from scheduler.manager import QueueManager, job_queue, shutdown_flag

#from shared.training.train import train
import importlib
import shared.training.train as training_module
from shared.training.training_status import TrainingStatus


training_function = None

def print_stage(stage, job, n=30):
    print("=" * n)
    print(f"[{stage}]")
    print(job)
    print("=" * n)

def worker_loop(queue):

    print("Worker started...")
    while not shutdown_flag.is_set():
        try:
            job = queue.get(timeout=1)
        except:
            continue    # no job, loop again
        
        try:
            importlib.reload(training_module)
            global training_function
            training_function = training_module.train
        except Exception as e:
            print('[ERROR] reloading training module')
            training_function = None

        if training_function is None:
            print("[ERROR] No training function available")
            continue
        
        print_stage("START", job)
        status, message = training_function(job)
        #status, message = training_module.train(job)
        
        if status == TrainingStatus.SUCCESS:
            print_stage("DONE", job)
        else:
            print_stage(status.value.upper(), message)
            
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
    


