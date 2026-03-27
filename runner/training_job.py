from dataclasses import dataclass, asdict, field
from enum import Enum

from shared.algorithm.algorithm_types import Algorithm
from runner.training_parser import build_parser_from_dataclass


@dataclass
class TrainingJob:
    algo: Algorithm = Algorithm.SAC
    
    lr: float = 1e-4
    discount_factor: float = 0.99

    net_arch: list[int] = field(default_factory=lambda: [64, 64, 64, 64])
    
    
    seed: int = 42
    parallel_env: int = 8
    total_steps: int = 1e6
    batch_steps: int = 2048
    
    device: str = 'cpu' # 'cuda' for gpu, and 'cpu' for cpu
    recording_frequency: int = 10
    verbose: int = 0
    
    def __str__(self):
        fields = vars(self)
        return "\n".join(f"{k}: {v}" for k, v in fields.items())
    

    def to_cli_args(self):
        args = []
        for k, v in asdict(self).items():
            if isinstance(v, list):
                # extend the list as separate arguments (for nargs="+")
                args.extend([f"--{k}"] + [str(i) for i in v])
            else:
                if isinstance(v, Enum):
                    v = v.value
                args.extend([f"--{k}", str(v)])
        
        return args
    

training_job_parser = build_parser_from_dataclass(TrainingJob)
