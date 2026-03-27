import json
from dataclasses import dataclass, asdict


@dataclass
class SchedulerConfig:
    name: str
    domain: str
    port: int
    auth_key: str

    @classmethod
    def from_dict(cls, data: dict) -> "SchedulerConfig":
        return cls(**data)

    @classmethod
    def from_json_file(cls, filepath: str) -> "SchedulerConfig":
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
        
        


