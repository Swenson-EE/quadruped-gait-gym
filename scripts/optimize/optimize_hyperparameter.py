from dataclasses import asdict, dataclass

import torch
import optuna
import os
import json

from shared.config.hyperparameters import Hyperparameters
from shared.utils import parse_args_to_dataclass

from .optimize import Optimizer, OptimizeArguments, optimize_arguments_parser





class HyperparameterOptimizer(Optimizer):   
    def __init__(self, args: OptimizeArguments):
        super().__init__(args)

        self.optimization_name = "hyperparameter_optimization"


        self.suggest_config = {
            "learning_rate": {
                "type": "float",
                "low": 1e-5,
                "high": 1e-3
            },
            "batch_size": {
                "type": "categorical",
                "choices": ["32", "64", "128", "256"]
            },
            "batch_steps": {
                "type": "categorical",
                "choices": ["512", "1024", "2048"]
            },
            "discount_factor": {
                "type": "float",
                "low": 0.9,
                "high": 0.99
            },
            "entropy_coeff": {
                "type": "float",
                "low": 0.0,
                "high": 0.02
            },
            "activation": {
                "type": "categorical",
                "choices": [
                    "tanh",
                    "relu",
                    "leaky-relu"
                ]
            }
        }

    def get_model_parameters(self, trial) -> Hyperparameters:
        suggested_parameters = self.suggest(Hyperparameters, trial, self.suggest_config)

        suggested_parameters["batch_size"] = int(suggested_parameters["batch_size"])
        suggested_parameters["batch_steps"] = int(suggested_parameters["batch_steps"])
        new_hyperparameters = Hyperparameters.from_dict(suggested_parameters)

        return new_hyperparameters


    def use_best_results(self, best_trial):
        hyperparameters = Hyperparameters.from_flat_dict(best_trial.params)
        hyperparameters.batch_size = int(best_trial.params["batch_size"])

        
        print('[BEST HYPERPARAMETERS]')
        print(hyperparameters)

        with open(os.path.join(self.SAVE_DIR, "hyperparameters.json"), "w") as f:
            json.dump(asdict(hyperparameters), f, indent = 4)


if __name__ == "__main__":
    args = parse_args_to_dataclass(optimize_arguments_parser, OptimizeArguments)
    

    optimizer = HyperparameterOptimizer(args=args)
    optimizer.run()

