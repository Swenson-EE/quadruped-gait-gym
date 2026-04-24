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

        self.LOW = 1e-4
        self.HIGH = 1e-2
        self.suggest_config = {
            "reward.secondary.*": {
                "type": "float",
                "low": 1e-4,
                "high": 1e-2
            },
            "reward.secondary_scale": {
                "type": "float",
                "low": 1e-4,
                "high": 1e-2
            },
            "penalty.secondary.*": {
                "type": "float",
                "low": 1e-4,
                "high": 1e-2
            },
            "penalty.secondary_scale": {
                "type": "float",
                "low": 1e-4,
                "high": 1e-2
            }
        }


    def get_weights(self, trial: optuna.Trial):
        weights: RewardWeights = super().get_weights(trial)

        suggested_weights = self.suggest(RewardWeights, trial, self.suggest_config)
        suggested_weights["reward"]["secondary"] = self.normalize(suggested_weights["reward"]["secondary"])
        suggested_weights["penalty"]["secondary"] = self.normalize(suggested_weights["penalty"]["secondary"])

        new_weights = RewardWeights.from_dict(suggested_weights | asdict(weights))
        
        return new_weights
    
    def use_best_results(self, best_trial):
        weights: RewardWeights = RewardWeights.from_flat_dict(best_trial.params)

        with open(os.path.join(self.SAVE_DIR, "weights.json"), "w") as f:
            json.dump(asdict(weights), f, indent = 4)

if __name__ == "__main__":
    args = parse_args_to_dataclass(optimize_arguments_parser, OptimizeArguments)
    args.optimization_name = "secondary_optimization"
    args.n_steps = 1e3
    args.n_trials = 1

    optimizer = SecondaryWeightOptimizer(args=args)
    optimizer.run()
