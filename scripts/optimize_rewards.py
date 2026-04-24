import json
import pprint

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.samplers import TPESampler
import optuna.visualization as vis

from dataclasses import dataclass, fields, asdict, is_dataclass

from shared.algorithm.algorithm_types import Algorithm
from shared.algorithm.algorithm_info import get_algorithm_class, get_algo_model
from shared.utils.dataclass_parser import build_parser_from_dataclass, parse_args_to_dataclass



LATERAL_WEIGHT = 0.9
Z_WEIGHT = 0.6

LOG_DIR = "saved/optimization"
EXCEL_PATH = "training_runs.xlsx"

os.makedirs(LOG_DIR, exist_ok=True)



@dataclass 
class OptimizeArguments:
    algorithm: Algorithm = Algorithm.PPO_C
    n_trials: int = 100
    n_jobs: int = 1

    n_threads: int = 4

optimize_arguments_parser = build_parser_from_dataclass(OptimizeArguments)


LOW = 0.001
HIGH = 1.0

@dataclass
class Weights:
    @dataclass
    class Reward:
        efficiency: float
        air_time: float
        forward_velocity: float
        # bent_joint: float

    @dataclass
    class Penalty:
        action_smooth: float
        height: float
        symmetry: float

        stability_angle: float
        stability_rate: float

        
        # lateral_velocity: float
        # z_velocity: float

        # lateral_movement: float

        # crawling: float
        # large_joint_variance: float
        # paw_clearance: float
        # slipping: float
        # paws_contacting: float
        # jitter_1st: float
        # jitter_2nd: float
        
        # joint_velocity: float
        # joint_acceleration: float
        # joint_jerk: float
        # joint_energy: float

    reward: Reward
    penalty: Penalty

    reward_scale: float
    penalty_scale: float

    @classmethod
    def from_flat_dict(cls, flat: dict) -> "Weights":
        flat = {
            k: (round(v, 5) if isinstance(v, float) else v) for k, v in flat.items() 
        }

        reward_fields = {f.name for f in fields(Weights.Reward)}
        penalty_fields = {f.name for f in fields(Weights.Penalty)}
        root_fields = {f.name for f in fields(Weights)} - {"reward", "penalty"}

        reward_data = {k: v for k, v in flat.items() if k in reward_fields}
        penalty_data = {k: v for k, v in flat.items() if k in penalty_fields}
        root_data = {k: v for k, v in flat.items() if k in root_fields}

        return cls(
            reward=Weights.Reward(**reward_data),
            penalty=Weights.Penalty(**penalty_data),
            **root_data
        )

    @classmethod
    def suggest(cls, trial: optuna.trial):
        def suggest_from_dataclass(dc_cls):
            result = {}

            for f in fields(dc_cls):
                name = f.name

                # Nested dataclass (Reward / Penalty)
                if is_dataclass(f.type):
                    result[name] = suggest_from_dataclass(f.type)
                # Leaf parameter
                else:
                    result[name] = trial.suggest_float(name, LOW, HIGH, log=True)

            return result

        return suggest_from_dataclass(cls)



def append_to_excel(df, path=os.path.join(LOG_DIR, EXCEL_PATH)):
    if not os.path.exists(path):
        df.to_excel(path, index=False)
    else:
        with pd.ExcelWriter(path, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
            existing = pd.read_excel(path)

            combined = pd.concat([existing, df], ignore_index=True)
            combined.to_excel(writer, index=False)


# ENVIRONMENT FACTORY
def make_env():
    environment_class, environment_parameters = get_algorithm_class(algorithm)

    model_class, model_parameters = get_algo_model(algorithm)

    return environment_class, environment_parameters, model_class, model_parameters


def normalize(d):
    vals = np.array(list(d.values()))
    vals = vals / np.sum(vals)
    return dict(zip(d.keys(), vals))

# WEIGHT SAMPLING
def sample_weights(trial):
    # weights = {
    #     "reward": {
    #         #"forward_movement": trial.suggest_float("forward_movement", 0.1, 3.0, log=True),
    #         "efficiency": trial.suggest_float("efficiency", 0.1, 5.0, log=True)
    #     },
    #     "penalty": {
    #         "lateral_movement": trial.suggest_float("lateral_movement", 0.1, 5.0, log=True),
    #         "z_movement": trial.suggest_float("z_movement", 0.1, 5.0, log=True),
    #         "crawling": trial.suggest_float("crawling", 0.1, 3.0, log=True),
    #         #"small_joint_variance": trial.suggest_float("small_joint_variance", 0.1, 3.0, log=True),
    #         "large_joint_variance": trial.suggest_float("large_joint_variance", 0.5, 5.0, log=True),
    #         "paw_clearance": trial.suggest_float("paw_clearance", 0.1, 3.0, log=True),
    #         "slipping": trial.suggest_float("slipping", 0.1, 3.0, log=True),
    #         "paws_contacting": trial.suggest_float("paws_contacting", 0.1, 3.0, log=True),
    #         "jitter_1st": trial.suggest_float("jitter_1st", 0.5, 5.0, log=True),
    #         "jitter_2nd": trial.suggest_float("jitter_2nd", 0.5, 5.0, log=True),
    #         "stability_angle": trial.suggest_float("stability_angle", 0.1, 3.0, log=True),
    #         "stability_rate": trial.suggest_float("stability_rate", 0.1, 3.0, log=True),
    #         "lateral_velocity": trial.suggest_float("lateral_velocity", 0.5, 10.0, log=True),
    #         "z_velocity": trial.suggest_float("z_velocity", 0.5, 10.0, log=True),
    #         "action_smooth": trial.suggest_float("action_smooth", 0.1, 3.0, log=True), 
    #         "joint_velocity": trial.suggest_float("joint_velocity", 0.1, 3.0, log=True),    
    #         "joint_acceleration": trial.suggest_float("joint_acceleration", 0.1, 3.0, log=True), 
    #         "joint_jerk": trial.suggest_float("joint_jerk", 0.1, 3.0, log=True), 
    #         "joint_energy": trial.suggest_float("joint_energy", 0.1, 3.0, log=True), 

    #     },
    #     "reward_scale": trial.suggest_float("reward_scale", 0.1, 3.0, log=True),
    #     "penalty_scale": trial.suggest_float("penalty_scale", 0.1, 3.0, log=True)
    # }
    weights = Weights.suggest(trial)

    weights["reward"] = normalize(weights["reward"])
    weights["penalty"] = normalize(weights["penalty"])

    with open("config/reward_weights.json") as file:
        data = json.load(file)

    return weights

# EVALUATION FUNCTION
def evaluate(model, env, n_episodes = 5):
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
def objective(trial):
    weights = sample_weights(trial)

    trial_id = f"trial_{trial.number}"
    monitor_dir = os.path.join(LOG_DIR, "monitors")
    os.makedirs(monitor_dir, exist_ok=True)

    environment_class, environment_parameters, model_class, model_parameters = make_env()
    

    monitor_file = os.path.join(monitor_dir, f"monitor_{trial_id}")
    env = Monitor(
        environment_class(parameters=environment_parameters(), weights = weights),
        filename=monitor_file
    )

    model = model_class(
        'MultiInputPolicy',
        env,
        seed=42,
        **model_parameters,
        policy_kwargs={
            "net_arch": [64, 64]
        },
        verbose=0
    )

    model.learn(total_timesteps=100_000)

    score = evaluate(model, env)

    # Load monitor data
    df = pd.read_csv(monitor_file + ".monitor.csv", skiprows=1)

    df["trial"] = trial.number
    #df["score"] = score
    forward, lateral, z = score
    print(forward, lateral, z)
    df["forward"] = forward
    df["lateral"] = lateral
    df["z"] = z

    for k, v in trial.params.items():
        df[k] = v

    # Append to excel
    append_to_excel(df)


    return score


def plot_trials():
    df = pd.read_excel(os.path.join(LOG_DIR, EXCEL_PATH))

    for trial_id, group in df.groupby("trial"):
        plt.plot(group["r"], label=f"Trial {trial_id}")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Curves per Trial")
    
    plt.savefig(os.path.join(LOG_DIR,"training_curves.png"), dpi=300, bbox_inches="tight")


def _score_trials(t: optuna.trial.FrozenTrial):
        forward, lateral, z = t.values
        return forward - LATERAL_WEIGHT*lateral - Z_WEIGHT*z

def plot(study: optuna.Study):
    # ===================================
    #           Overall plots
    # ===================================
    os.makedirs(os.path.join(LOG_DIR, "overall"), exist_ok=True)


    target_name = "Overall Score"
    fig = vis.plot_pareto_front(
        study,
        #target=lambda t: t.values[0] - LATERAL_WEIGHT*t.values[1] - Z_WEIGHT*t.values[2],
        target_names=["forward", "lateral", "z"]
    )
    fig.write_image(os.path.join(LOG_DIR, "overall", f"optimization_history_overall.png"))

    fig = vis.plot_param_importances(
        study,
        target=lambda t: t.values[0] - LATERAL_WEIGHT*t.values[1] - Z_WEIGHT*t.values[2],
        target_name=target_name
    )
    fig.update_layout(title="Overall Hyperparameter Importance")
    fig.write_image(os.path.join(LOG_DIR, "overall", f"param_importances_overall.png"))

    # Parallel coordinate
    fig = vis.plot_optimization_history(
        study,
        target=lambda t: t.values[0] - LATERAL_WEIGHT*t.values[1] - Z_WEIGHT*t.values[2],
        target_name=target_name
    )
    #fig.update_layout(title="Overall Optimization History")
    fig.write_image(os.path.join(LOG_DIR, "overall", f"optimization_history_overall.png"))



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
        fig.write_image(os.path.join(LOG_DIR, f"pareto_{name}.png"))



    # ===================================
    #           Individual plots
    # ===================================
    
    def get_top_trials(study, index, top_n: 50):
        trials = [t for t in study.trials if t.values is not None]
        trials = sorted(trials, key=lambda t: t.values[index], reverse=True)
        return trials[:top_n]
    
    trial_values = {
        0: "forward",
        1: "lateral",
        2: "z"
    }


    for index, name in trial_values.items():
        trial_dir = os.path.join(LOG_DIR, name)
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

def save_weights(study: optuna.Study):
    best_trial = max(study.best_trials, key=_score_trials)

    weights: Weights = Weights.from_flat_dict(best_trial.params) 
    with open(os.path.join(LOG_DIR, "reward_weights.json"), "w") as f:
        json.dump(asdict(weights), f, indent=4)


# MAIN LOOP
def main(args: OptimizeArguments):
    study = optuna.create_study(
        study_name="reward_optimization",
        storage=f"sqlite:///{os.path.join(LOG_DIR, 'optuna_study.db')}",
        load_if_exists=True,
        directions=["maximize", "minimize", "minimize"],
        sampler=TPESampler(multivariate=True),
        pruner=optuna.pruners.MedianPruner()
    )

    # study.optimize(
    #     objective,
    #     n_trials=args.n_trials,
    #     n_jobs=args.n_jobs # increase for parallel jobs
    # )
    
    
    best_trial = max(study.best_trials, key=_score_trials)
    print("\n", "="*5, " [BEST RESULT] ", "="*5)
    print("Score:", best_trial.values)
    print("Weights")
    pprint.pprint(best_trial.params, width=50)

    save_weights(study)

    # plot(study)
    

        # fig = vis.plot_param_importances(
        #     study,
        #     target=lambda t: t.values[index],
        #     target_name=target_name
        # )
        # fig.update_layout(title=f"{name.capitalize} Hyperparameter Importances")
        # fig.write_image(os.path.join(trial_dir, f"optimization_history_{name}.png"))

        # fig = vis.plot_optimization_history(
        #     study,
        #     target=lambda t: t.values[index],
        #     target_name=target_name
        # )
        # fig.update_layout(title=f"")
        # fig.write_image(os.path.join(trial_dir, f"param_importances_{name}.png"))

        # # Parallel coordinate
        # fig = vis.plot_parallel_coordinate(
        #     study,
        #     target=lambda t: t.values[index],
        #     target_name=target_name
        # )
        # fig.write_image(os.path.join(trial_dir, f"parallel_coordinates_{name}.png"))

    
    
    

    plot_trials()



if __name__ == "__main__":
    args = parse_args_to_dataclass(optimize_arguments_parser, OptimizeArguments)
    
    import torch
    torch.set_num_threads(args.n_threads)

    print("#"*10, f"[TRAINING {args.algorithm.value.upper()}]", "#"*10)
    print(args.n_trials, "trials")
    print(args.n_jobs, "jobs")
    print('\n')

    global algorithm
    algorithm = args.algorithm

    main(args)

