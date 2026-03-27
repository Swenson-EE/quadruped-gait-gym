from multiprocessing.managers import BaseManager
from queue import Queue
import threading

job_queue = Queue()
shutdown_flag = threading.Event()

class QueueManager(BaseManager):
    pass

# Expose the queue
QueueManager.register('get_queue', callable=lambda: job_queue)

# Expose a callable to set the shutdown flag
def shutdown_worker():
    shutdown_flag.set()
    return "shutdown signal sent"

QueueManager.register('quit', callable=shutdown_worker)


