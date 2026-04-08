import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import optuna
import optuna.visualization as vis

from dataclasses import dataclass

from shared.algorithm.algorithm_types import Algorithm
from shared.algorithm.algorithm_info import get_algorithm_class, get_algo_model
from shared.utils.dataclass_parser import build_parser_from_dataclass, parse_args_to_dataclass



LOG_DIR = "saved/optimization"
EXCEL_PATH = "training_runs.xlsx"

os.makedirs(LOG_DIR, exist_ok=True)



@dataclass 
class OptimizeArguments:
    algorithm: Algorithm = Algorithm.PPO_C
    n_trials: int = 100
    n_jobs: int = 1

optimize_arguments_parser = build_parser_from_dataclass(OptimizeArguments)


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
    environment_class = get_algorithm_class(algorithm)

    model_class, model_parameters = get_algo_model(algorithm)

    return environment_class, model_class, model_parameters


def normalize(d):
    vals = np.array(list(d.values()))
    vals = vals / np.sum(vals)
    return dict(zip(d.keys(), vals))

# WEIGHT SAMPLING
def sample_weights(trial):
    weights = {
        "reward": {
            #"forward_movement": trial.suggest_float("forward_movement", 0.1, 3.0, log=True),
            "efficiency": trial.suggest_float("efficiency", 0.1, 5.0, log=True)
        },
        "penalty": {
            "lateral_movement": trial.suggest_float("lateral_movement", 0.1, 5.0, log=True),
            "z_movement": trial.suggest_float("z_movement", 0.1, 5.0, log=True),
            "crawling": trial.suggest_float("crawling", 0.1, 3.0, log=True),
            "small_joint_variance": trial.suggest_float("small_joint_variance", 0.1, 3.0, log=True),
            "large_joint_variance": trial.suggest_float("large_joint_variance", 0.5, 5.0, log=True),
            "paw_clearance": trial.suggest_float("paw_clearance", 0.1, 3.0, log=True),
            "slipping": trial.suggest_float("slipping", 0.1, 3.0, log=True),
            "paws_contacting": trial.suggest_float("paws_contacting", 0.1, 3.0, log=True),
            "jitter_1st": trial.suggest_float("jitter_1st", 0.5, 5.0, log=True),
            "jitter_2nd": trial.suggest_float("jitter_2nd", 0.5, 5.0, log=True),
            "stability_angle": trial.suggest_float("stability_angle", 0.1, 3.0, log=True),
            "stability_rate": trial.suggest_float("stability_rate", 0.1, 3.0, log=True),
            "lateral_velocity": trial.suggest_float("lateral_velocity", 0.5, 10.0, log=True),
            "z_velocity": trial.suggest_float("z_velocity", 0.5, 10.0, log=True)    
        },
        "reward_scale": trial.suggest_float("reward_scale", 0.1, 3.0, log=True),
        "penalty_scale": trial.suggest_float("penalty_scale", 0.1, 3.0, log=True)
    }

    weights["reward"] = normalize(weights["reward"])
    weights["penalty"] = normalize(weights["penalty"])

    return weights

# EVALUATION FUNCTION
def evaluate(model, env, n_episodes = 5):
    total_reward = 0.0
    success_count = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            done = terminated or truncated

        total_reward += ep_reward

    # TODO: Success metric
    if info.get("is_success", False):
        success_count += 1

    # Prefer success rate if available
    if success_count > 0:
        return success_count / n_episodes

    return total_reward / n_episodes



# OPTUNA OBJECTIVE
def objective(trial):
    weights = sample_weights(trial)

    trial_id = f"trial_{trial.number}"
    monitor_dir = os.path.join(LOG_DIR, "monitors")
    os.makedirs(monitor_dir, exist_ok=True)

    environment_class, model_class, model_parameters = make_env()


    monitor_file = os.path.join(monitor_dir, f"monitor_{trial_id}")
    env = Monitor(
        environment_class(weights = weights),
        filename=monitor_file
    )

    model = model_class(
        'MultiInputPolicy',
        env,
        seed=42,
        **model_parameters,
        verbose=0
    )

    model.learn(total_timesteps=100_000)

    score = evaluate(model, env)

    # Load monitor data
    df = pd.read_csv(monitor_file + ".monitor.csv", skiprows=1)

    df["trial"] = trial.number
    df["score"] = score

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
    plt.legend()
    
    plt.savefig(os.path.join(LOG_DIR,"training_curves.png"), dpi=300, bbox_inches="tight")

    #plt.show()


# MAIN LOOP
def main(args: OptimizeArguments):
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs # increase for parallel jobs
    )

    print("\n", "="*5, " [BEST RESULT] ", "="*5)
    print("Score:", study.best_value)
    print("Weights:", study.best_params)

    # Optimization history
    fig = vis.plot_optimization_history(study)
    fig.write_image(os.path.join(LOG_DIR, "optimization_history.png"))

    # Parameter importance
    fig = vis.plot_param_importances(study)
    fig.write_image(os.path.join(LOG_DIR, "param_importances.png"))

    # Parallel coordinate
    fig = vis.plot_parallel_coordinate(study)
    fig.write_image(os.path.join(LOG_DIR, "parallel_coordinates.png"))


    plot_trials()



if __name__ == "__main__":
    args = parse_args_to_dataclass(optimize_arguments_parser, OptimizeArguments)
    
    print("#"*10, f"[TRAINING {args.algorithm.value.upper()}]", "#"*10)
    print(args.n_trials, "trials")
    print(args.n_jobs, "jobs")
    print('\n')

    global algorithm
    algorithm = args.algorithm

    main(args)

