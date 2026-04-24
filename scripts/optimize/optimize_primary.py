from dataclasses import asdict

import optuna
import os
import json

from shared.rewards import RewardWeights
from shared.utils import parse_args_to_dataclass

from .optimize import Optimizer, OptimizeArguments, optimize_arguments_parser



class PrimaryWeightOptimizer(Optimizer):

    def get_weights(self, trial: optuna.Trial):
        weights: RewardWeights = super().get_weights(trial)

        selected_fields = [
            "reward.primary.*",
            "reward.primary_scale",

            "penalty.primary.*",
            "penalty.primary_scale"
        ]
        suggested_weights = self.suggest(RewardWeights, trial, selected_fields)
        suggested_weights["reward"]["primary"] = self.normalize(suggested_weights["reward"]["primary"])
        suggested_weights["penalty"]["primary"] = self.normalize(suggested_weights["penalty"]["primary"])


        new_weights = RewardWeights.from_dict(asdict(weights) | suggested_weights)

        return new_weights
    
    def use_best_results(self, best_trial):
        weights: RewardWeights = RewardWeights.from_flat_dict(best_trial.params)

        with open(os.path.join(self.SAVE_DIR, "weights.json"), "w") as f:
            json.dump(asdict(weights), f, indent = 4)


if __name__ == "__main__":
    args = parse_args_to_dataclass(optimize_arguments_parser, OptimizeArguments)
    args.optimization_name = "primary_optimization"

    optimizer = PrimaryWeightOptimizer(args=args)
    optimizer.run()

