import time

from scheduler.base import SchedulerConfig
from scheduler.manager import QueueManager, job_queue, shutdown_flag

from train import train
from runner.process_factory import ProcessFactory



def train_job(job):
    print(f"[TRAINING] {job}")
    train(job)
    #time.sleep(1)

def worker_loop(queue):
    factory = ProcessFactory(script='train.py')

    print("Worker started...")
    while not shutdown_flag.is_set():
        try:
            job = queue.get(timeout=1)
        except:
            continue    # no job, loop again
        
        print(f"[START] {job}")
        factory.run(job, blocking=True)
        print(f"[DONE] {job}")

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
    


