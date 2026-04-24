from dataclasses import dataclass
from dacite import from_dict
import json

from shared.utils import build_from_flat





@dataclass
class ConfigCls:
    @classmethod
    def from_json_file(cls, filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)

        return from_dict(data_class=cls, data=data)

    @classmethod
    def from_dict(cls, data: dict):
        return from_dict(data_class=cls, data=data)

    @classmethod
    def from_flat_dict(cls, flat: dict):
        with open("config/reward_weights.json") as f:
            defaults = json.load(f)

        return build_from_flat(cls, flat, defaults=defaults)
    
