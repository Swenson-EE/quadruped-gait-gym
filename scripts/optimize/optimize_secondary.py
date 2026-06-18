from dataclasses import asdict

import optuna
import os
import json

from shared.rewards import RewardWeights
from shared.utils import parse_args_to_dataclass

from .optimize import Optimizer, OptimizeArguments, optimize_arguments_parser



class SecondaryWeightOptimizer(Optimizer):
    def __init__(self, args: OptimizeArguments):
        super().__init__(args)

        self.optimization_name = "secondary_optimization"


        self.LOW = 1e-4
        self.HIGH = 1e-2
        self.suggest_config = {
            "reward.secondary.*": {
                "type": "float",
                "low": 0.01,
                "high": 1.0
            },
            "reward.secondary_scale": {
                "type": "float",
                "low": 1e-2,
                "high": 1e-1
            },
            "penalty.secondary.*": {
                "type": "float",
                "low": 0.01,
                "high": 1.0
            },
            "penalty.secondary_scale": {
                "type": "float",
                "low": 1e-2,
                "high": 1e-1
            }
        }


    def get_weights(self, trial: optuna.Trial):
        weights: RewardWeights = super().get_weights(trial)

        suggested_weights = self.suggest(RewardWeights, trial, self.suggest_config)

        suggested_weights["reward"]["secondary"] = self.normalize(suggested_weights["reward"]["secondary"])
        suggested_weights["penalty"]["secondary"] = self.normalize(suggested_weights["penalty"]["secondary"])

        combined = {
            "reward": {
                "primary": weights.reward.primary,
                "primary_scale": weights.reward.primary_scale,
                "secondary": suggested_weights["reward"]["secondary"],
                "secondary_scale": suggested_weights["reward"]["secondary_scale"]
            },
            "penalty": {
                "primary": weights.penalty.primary,
                "primary_scale": weights.penalty.primary_scale,
                "secondary": suggested_weights["penalty"]["secondary"],
                "secondary_scale": suggested_weights["penalty"]["secondary_scale"]
            }
        }

        new_weights = RewardWeights.from_dict(combined)
        return new_weights
    
    def use_best_results(self, best_trial):
        weights: RewardWeights = RewardWeights.from_flat_dict(best_trial.params)
        
        weights = asdict(weights)

        weights["reward"]["secondary"] = self.normalize(weights["reward"]["secondary"])
        weights["penalty"]["secondary"] = self.normalize(weights["penalty"]["secondary"])

        weights: RewardWeights = RewardWeights.from_dict(weights)

        with open(os.path.join(self.SAVE_DIR, "weights.json"), "w") as f:
            json.dump(asdict(weights), f, indent = 4)

if __name__ == "__main__":
    args = parse_args_to_dataclass(optimize_arguments_parser, OptimizeArguments)
    

    optimizer = SecondaryWeightOptimizer(args=args)
    optimizer.run()
