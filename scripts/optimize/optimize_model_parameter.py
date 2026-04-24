from dataclasses import asdict, dataclass

import optuna
import os
import json

from shared.config.config_cls import ConfigCls
from shared.config.environment_parameters import EnvironmentParameters
from shared.utils import parse_args_to_dataclass

from .optimize import Optimizer, OptimizeArguments, optimize_arguments_parser





class ModelParameterOptimizer(Optimizer):   
    def __init__(self, args: OptimizeArguments):
        super().__init__(args)

        self.suggest_config = {
            "joint_max": {
                "type": "int",
                "low": 60,
                "high": 120
            },
            "joint_delta": {
                "type": "int",
                "low": 5,
                "high": 90
            },
            "phase_rate": {
                "type": "float",
                "low": 0.1,
                "high": 10.0
            },
            "length_joint_history": {
                "type": "int",
                "low": 5,
                "high": 100
            },
            "episode_length": {
                "type": "int",
                "low": 100,
                "high": 2000
            }
        }

    def get_environment_parameters(self, trial):
        suggested_parameters = self.suggest(EnvironmentParameters, trial, self.suggest_config)
        suggested_parameters["joint_min"] = -suggested_parameters["joint_max"]
        suggested_parameters["total_length"] = int(self.args.n_steps)

        return EnvironmentParameters.from_dict(suggested_parameters)

    def use_best_results(self, best_trial):
        model_parameters = EnvironmentParameters.from_flat_dict(best_trial.params)
        
        print('[BEST MODEL PARAMETERS]')
        print(model_parameters)

        with open(os.path.join(self.SAVE_DIR, "model_parameters.json"), "w") as f:
            json.dump(asdict(model_parameters), f, indent = 4)

if __name__ == "__main__":
    args = parse_args_to_dataclass(optimize_arguments_parser, OptimizeArguments)
    args.optimization_name = "model_optimization"
    args.n_steps = 1e3
    args.n_trials = 2

    optimizer = ModelParameterOptimizer(args=args)
    optimizer.run()

