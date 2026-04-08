import gymnasium as gym
import numpy as np
import optuna

from dataclasses import dataclass

from shared.algorithm.algorithm_types import Algorithm
from shared.algorithm.algorithm_info import get_algorithm_class, get_algo_model
from shared.utils.dataclass_parser import build_parser_from_dataclass, parse_args_to_dataclass




@dataclass 
class OptimizeArguments:
    algorithm: Algorithm = Algorithm.PPO_C

optimize_arguments_parser = build_parser_from_dataclass(OptimizeArguments)


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
            "forward_movement": trial.suggest_float("forward_movement", 0.1, 3.0, log=True),
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

    environment_class, model_class, model_parameters = make_env()

    env = environment_class(weights = weights)

    model = model_class(
        'MultiInputPolicy',
        env,
        **model_parameters,
        verbose=0
    )

    model.learn(total_timesteps=100_000)

    return evaluate(model, env)


    


# MAIN LOOP
def main():
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(
        objective,
        n_trials=50,
        n_jobs=1 # increase for parallel jobs
    )

    print("\n", "="*5, " [BEST RESULT] ", "="*5)
    print("Score:", study.best_value)
    print("Weights:", study.best_params)


if __name__ == "__main__":
    args = parse_args_to_dataclass(optimize_arguments_parser, OptimizeArguments)
    
    global algorithm
    algorithm = args.algorithm

    main()

