from dataclasses import dataclass, fields, asdict, is_dataclass
import json

from shared.algorithm.algorithm_types import Algorithm

from shared.algorithm.algorithm_info import get_algorithm_class, get_algo_model
from shared.config.hyperparameters import Hyperparameters
from shared.rewards import RewardWeights
from shared.utils.dataclass_parser import build_parser_from_dataclass, parse_args_to_dataclass

import os
import pandas as pd
import numpy as np

from stable_baselines3.common.monitor import Monitor

import optuna
from optuna.samplers import TPESampler
import optuna.visualization as vis
import matplotlib.pyplot as plt





# SAVE_DIR = "saved/optimization"
# EXCEL_PATH = "training_runs.xlsx"

# global algorithm



LOG_DIR = "saved/optimization"
EXCEL_PATH = "training_runs.xlsx"

@dataclass 
class OptimizeArguments:
    algorithm: Algorithm = Algorithm.PPO_C
    n_trials: int = 100
    n_jobs: int = 1

    n_threads: int = 4

    n_steps: int = 1e6

    optimization_name: str = "optimization"

optimize_arguments_parser = build_parser_from_dataclass(OptimizeArguments)


class Optimizer:

    def __init__(self, args: OptimizeArguments):
        self.args = args

        self.LATERAL_WEIGHT = 0.9
        self.Z_WEIGHT = 0.6

        #self.LOW = 0.001
        #self.HIGH = 1.0
        
        self.suggest_config = {}

        self.SAVE_DIR = os.path.join(LOG_DIR, args.optimization_name)

        os.makedirs(self.SAVE_DIR, exist_ok=True)


    # ENVIRONMENT FACTORY
    def make_env(self):
        environment_class, environment_parameters = get_algorithm_class(self.args.algorithm)
        model_class, model_parameters = get_algo_model(self.args.algorithm)
        return environment_class, environment_parameters, model_class, model_parameters
    
    # EVALUATION FUNCTION
    def evaluate(self, model, env, n_episodes = 5):
        metrics = []

        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False

            forward = lateral = z = 0.0
            steps = 0


            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                start_pos = info["start_position"]
                end_pos = info["end_position"]

                dx, dy, dz = (end_pos - start_pos)
                
                forward += dx
                lateral += abs(dy)
                z += abs(dz)

                steps += 1

                done = terminated or truncated

            metrics.append((
                forward / steps,
                lateral / steps,
                z / steps
            ))

        mean_forward = np.mean([m[0] for m in metrics])
        mean_lateral = np.mean([m[1] for m in metrics])
        mean_z = np.mean([m[2] for m in metrics])

        return mean_forward, mean_lateral, mean_z

    # OPTUNA OBJECTIVE
    def objective(self, trial):
        trial_id = f"trial_{trial.number}"
        monitor_dir = os.path.join(self.SAVE_DIR, "monitors")
        os.makedirs(monitor_dir, exist_ok=True)

        environment_class, environment_parameters_cls, model_class, model_parameters = self.make_env()

        weights = self.get_weights(trial)
        env_params = self.get_environment_parameters(trial)
        hyperparameters = self.get_model_parameters(trial)
        #model_parameters = model_parameters | self.get_model_parameters(trial)

        #print(f'test\n{environment_parameters_cls(**env_params)}\n\n')

        monitor_file = os.path.join(monitor_dir, f"monitor_{trial_id}")
        env = Monitor(
            environment_class(parameters=environment_parameters_cls(**env_params), weights = weights),
            filename=monitor_file
        )
        
        #print(f"weights\n{weights}\n")
        #print(f"env_params\n{env_params}\n")
        #print(f"model_params\n{model_parameters}\n")


        model = model_class(
            'MultiInputPolicy',
            env,
            seed=42,
            policy_kwargs={
                "net_arch": [64, 64]
            },
            verbose=0,
            #**model_parameters,            
        )

        model.learn(total_timesteps=self.args.n_steps)

        score = self.evaluate(model, env)

        # Load monitor data
        df = pd.read_csv(monitor_file + ".monitor.csv", skiprows=1)

        df["trial"] = trial.number
        forward, lateral, z = score

        df["forward"] = forward
        df["lateral"] = lateral
        df["z"] = z

        for k, v in trial.params.items():
            df[k] = v

        # Append to excel
        self.append_to_excel(df)


        return score

    def append_to_excel(self, df):
        path = os.path.join(self.SAVE_DIR, EXCEL_PATH)

        if not os.path.exists(path):
            df.to_excel(path, index=False)
        else:
            with pd.ExcelWriter(path, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
                existing = pd.read_excel(path)

                combined = pd.concat([existing, df], ignore_index=True)
                combined.to_excel(writer, index=False)

    def plot_trials(self):
        df = pd.read_excel(os.path.join(self.SAVE_DIR, EXCEL_PATH))

        for trial_id, group in df.groupby("trial"):
            plt.plot(group["r"], label=f"Trial {trial_id}")

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Curves per Trial")
        
        plt.savefig(os.path.join(self.SAVE_DIR,"training_curves.png"), dpi=300, bbox_inches="tight")

    from dataclasses import fields, is_dataclass


    def normalize(self, d):
        vals = np.array(list(d.values()))
        vals = vals / np.sum(vals)
        return dict(zip(d.keys(), vals))

    # def suggest(self, cls, trial, patterns: list[str]):
    #     from dataclasses import fields, is_dataclass

    #     import fnmatch

    #     def is_selected(full_name: str, patterns: list[str]) -> bool:
    #         return any(fnmatch.fnmatch(full_name, p) for p in patterns)
        
        
    #     def suggest_from_dataclass(dc_cls, prefix=""):
    #         result = {}

    #         for f in fields(dc_cls):
    #             name = f.name
    #             full_name = f"{prefix}.{name}" if prefix else name

    #             if is_dataclass(f.type):
    #                 result[name] = suggest_from_dataclass(f.type, full_name)
    #             else:
    #                 if is_selected(full_name, patterns):
    #                     result[name] = trial.suggest_float(
    #                         full_name, self.LOW, self.HIGH, log=True
    #                     )
    #                 else:
    #                     result[name] = 0.0

    #         return result

    #     return suggest_from_dataclass(cls)
    
    def suggest(self, cls, trial, suggest_config: dict):
        from dataclasses import fields, is_dataclass

        import fnmatch

        # -----------------------------
        # Pattern matching (with priority)
        # -----------------------------
        def find_config(name: str):
            matches = []

            for pattern, cfg in suggest_config.items():
                if fnmatch.fnmatch(name, pattern):
                    matches.append((pattern, cfg))

            if not matches:
                return None

            # More specific pattern wins (longest string)
            matches.sort(key=lambda x: len(x[0]), reverse=True)
            return matches[0][1]

        # -----------------------------
        # Suggest value based on config
        # -----------------------------
        def suggest_value(name: str, cfg: dict):
            t = cfg["type"]

            if t == "float":
                return trial.suggest_float(
                    name,
                    cfg["low"],
                    cfg["high"],
                    log=cfg.get("log", False),
                    step=cfg.get("step", None)
                )

            elif t == "int":
                return trial.suggest_int(
                    name,
                    cfg["low"],
                    cfg["high"],
                    step=cfg.get("step", 1),
                    log=cfg.get("log", False)
                )

            elif t == "categorical":
                return trial.suggest_categorical(
                    name,
                    cfg["choices"]
                )

            else:
                raise ValueError(f"Unsupported suggest type: {t}")

        # -----------------------------
        # Recursive builder
        # -----------------------------
        def recurse(dc_cls, prefix=""):
            instance = dc_cls()  # get defaults
            result = {}

            for f in fields(dc_cls):
                name = f.name
                full_name = f"{prefix}.{name}" if prefix else name

                if is_dataclass(f.type):
                    result[name] = recurse(f.type, full_name)
                else:
                    cfg = find_config(full_name)

                    if cfg:
                        result[name] = suggest_value(full_name, cfg)
                    else:
                        # fallback to default value
                        result[name] = getattr(instance, name)

            return result

        return recurse(cls)



    def get_weights(self, trial: optuna.Trial) -> RewardWeights:
        weights: RewardWeights = RewardWeights.from_json_file("config/reward_weights.json")

        return weights
    
    def get_environment_parameters(self, trial: optuna.Trial):
        return {}

    def get_model_parameters(self, trial: optuna.Trial) -> Hyperparameters:
        return Hyperparameters()
    

    def print_best_results(self, best_trial: optuna.trial.FrozenTrial):
        print("\n", "="*5, " [BEST RESULT] ", "="*5)
        print("Score:", best_trial.values)
        print(f"[{self.args.optimization_name}]\n", best_trial.params, "\n")

    def use_best_results(self, best_trial: optuna.trial.FrozenTrial):
        pass

    def plot_study(self, study: optuna.Study):
        # ===================================
        #           Overall plots
        # ===================================
        os.makedirs(os.path.join(self.SAVE_DIR, "overall"), exist_ok=True)

        target_name = "Overall Score"
        fig = vis.plot_pareto_front(
            study,
            target_names=["forward", "lateral", "z"]
        )
        fig.write_image(os.path.join(self.SAVE_DIR, "overall", f"optimization_history_overall.png"))


        trial_target = lambda t: t.values[0] - (self.LATERAL_WEIGHT*t.values[1]) - (self.Z_WEIGHT*t.values[2])

        fig = vis.plot_param_importances(
            study,
            target=trial_target,
            target_name=target_name
        )
        fig.update_layout(title="Overall Hyperparameter Importance")
        fig.write_image(os.path.join(self.SAVE_DIR, "overall", f"param_importances_overall.png"))

        # Parallel coordinate
        fig = vis.plot_optimization_history(
            study,
            target=trial_target,
            target_name=target_name
        )
        fig.update_layout(title="Overall Optimization History")
        fig.write_image(os.path.join(self.SAVE_DIR, "overall", f"optimization_history_overall.png"))


        pairs = [
            (0, 1, "forward_vs_lateral"),
            (0, 2, "forward_vs_z"),
            (1, 2, "lateral_vs_z")
        ]

        for i, j, name in pairs:
            fig = vis.plot_pareto_front(
                study,
                target_names=["Forward, Lateral", "Z"],
                targets=lambda t: (t.values[i], t.values[j])
            )
            fig.update_layout(title=name.replace("_", " ").title())
            fig.write_image(os.path.join(self.SAVE_DIR, f"pareto_{name}.png"))



        # ===================================
        #           Individual plots
        # ===================================        
        trial_values = {
            0: "forward",
            1: "lateral",
            2: "z"
        }


        for index, name in trial_values.items():
            trial_dir = os.path.join(self.SAVE_DIR, name)
            os.makedirs(trial_dir, exist_ok=True)

            target_name = f"{name.capitalize()} Distance"

            fig = vis.plot_param_importances(
                study,
                target=lambda t: t.values[index],
                target_name=target_name
            )
            fig.update_layout(title=f"{name.capitalize()} Hyperparameter Importances")
            fig.write_image(os.path.join(trial_dir, f"param_importance_{name}.png"))


            fig = vis.plot_slice(
                study,
                target=lambda t: t.values[index],
                target_name=target_name
            )
            fig.update_layout(title=f"{name.capitalize()} Slice Plot")
            fig.write_image(os.path.join(trial_dir, f"slice_{name}.png"))


            fig = vis.plot_contour(
                study,
                target=lambda t: t.values[index],
                target_name=target_name
            )
            fig.update_layout(title=f"{name.capitalize()} Contour Plot")
            fig.write_image(os.path.join(trial_dir, f"contour_{name}.png"))



            fig = vis.plot_parallel_coordinate(
                study,
                params=None,
                target=lambda t: t.values[index],
                target_name=target_name
            )
            fig.update_layout(title=f"{name.capitalize()} Parallel Coordinates")
            fig.write_image(os.path.join(trial_dir, f"parallel_{name}.png"))


    def create_study(self) -> optuna.Study:
        return optuna.create_study(
            study_name=self.args.optimization_name,
            storage=f"sqlite:///{os.path.join(self.SAVE_DIR, f'{self.args.optimization_name}_study.db')}",
            load_if_exists=True,
            directions=["maximize", "minimize", "minimize"],
            sampler=TPESampler(multivariate=True),
            pruner=optuna.pruners.MedianPruner()
        )
    
    def _score_trials(self, t: optuna.trial.FrozenTrial):
            forward, lateral, z = t.values
            return forward - self.LATERAL_WEIGHT*lateral - self.Z_WEIGHT*z

    def view(self):
        study = self.create_study()
        best_trial = max(study.best_trials, key=self._score_trials)
        self.print_best_results(best_trial)


    def run(self):
        import torch
        torch.set_num_threads(self.args.n_threads)

        print("#"*10, f"[TRAINING {self.args.algorithm.value.upper()}]", "#"*10)
        print(self.args.n_trials, "trials")
        print(self.args.n_jobs, "jobs")
        print('\n')

        study = self.create_study()

        study.optimize(
            self.objective,
            n_trials=self.args.n_trials,
            n_jobs=self.args.n_jobs # increase for parallel jobs
        )
        
        best_trial = max(study.best_trials, key=self._score_trials)
        self.print_best_results(best_trial)
        self.use_best_results(best_trial)
        self.plot_study(study)

    