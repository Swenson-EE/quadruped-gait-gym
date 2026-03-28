import subprocess
import sys
import os

from runner.training_job import TrainingJob


class ProcessFactory:
    def __init__(self, script="shared.training.train"):
        self.script = script
        self.python_exec = sys.executable

    def build(self, job: TrainingJob):
        cmd = [
            self.python_exec,
            "-m",
            self.script,
            *job.to_cli_args()
        ]
        return cmd
    
    def run(self, job: TrainingJob, blocking=True):
        cmd = self.build(job)
        # Copy the current environment to inherit venv
        env = os.environ.copy()

        if blocking:
            return subprocess.run(cmd, env=env)
        else:
            return subprocess.Popen(cmd, env=env)
