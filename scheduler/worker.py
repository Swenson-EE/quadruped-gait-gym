import time

from scheduler.base import SchedulerConfig
from scheduler.manager import QueueManager, job_queue, shutdown_flag

from shared.training.train import train
from runner.process_factory import ProcessFactory

def print_stage(stage, job, n=30):
    print("=" * n)
    print(f"[{stage}]")
    print(job)
    print("=" * n)

def worker_loop(queue):
    factory = ProcessFactory(script='shared.training.train')

    print("Worker started...")
    while not shutdown_flag.is_set():
        try:
            job = queue.get(timeout=1)
        except:
            continue    # no job, loop again
        
        print_stage("START", job)
        factory.run(job, blocking=True)
        print_stage("DONE", job)

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
    


