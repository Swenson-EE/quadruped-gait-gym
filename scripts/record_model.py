from dataclasses import dataclass, field
import json
import time

from gymnasium.wrappers import RecordVideo
import imageio
import mujoco.viewer
import numpy as np

from shared.config.environment_parameters import EnvironmentParameters
from shared.rewards.rewards import RewardWeights
from shared.utils.dataclass_parser import build_parser_from_dataclass
from shared.algorithm.algorithm_info import get_algo_environment, get_algo_model, get_algorithm_class
from shared.algorithm.algorithm_types import Algorithm
from shared.checkpoints.checkpoints_names import get_checkpoint, get_latest_checkpoint


@dataclass
class TrainedModel:
    algo: Algorithm = Algorithm.PPO_C
    net_arch: list[int] = field(default_factory=lambda: [64, 64, 64, 64])
    n: int = None





if __name__ == "__main__":
    np.set_printoptions(linewidth=120)

    trained_model_parser = build_parser_from_dataclass(TrainedModel)
    args = trained_model_parser.parse_args()

    trained_model = TrainedModel(**vars(args))
    

    checkpoint_name: str = None
    if trained_model.n is None:
        checkpoint_name = get_latest_checkpoint(
            algo=trained_model.algo,
            # Any additional metadata for checkpoint naming
            layers=trained_model.net_arch
        )
    else:
        checkpoint_name = get_checkpoint(
            algo=trained_model.algo,
            n=trained_model.n,
            # Any additional metadata for checkpoint naming
            layers=trained_model.net_arch
        )
    
    

    if checkpoint_name is None:
        print("No checkpoints available")
        exit()

    print(f"Loading checkpoint {checkpoint_name}")

    weights = {}
    try:
        with open("config/reward_weights.json") as file:
            weights = json.load(file)
            print("weight config found")
    except FileNotFoundError:
        print("No weights config found; using defaults")
    weights = RewardWeights.from_dict(weights)


    environment_parameters = {}
    try:
        with open("config/environment_parameters.json") as file:
            environment_parameters = json.load(file)
            print('environment config found')
    except FileNotFoundError:
        print("no environment config found; using defaults")

    environment_parameters = EnvironmentParameters.from_dict(environment_parameters)


    AlgoEnvClass = get_algorithm_class(trained_model.algo)

    env = AlgoEnvClass(
        parameters=environment_parameters,
        weights=weights
    )
    if env is None:
        print("No env instantiated")
        exit()
    # env = get_algo_environment(trained_model.algo, weights=weights, parameters=environment_parameters)
    # if env is None:
    #     print("No env instantiated")
    #     exit()


    ModelClass = get_algo_model(trained_model.algo)
    if ModelClass is None:
        print('No model instantiated')
        exit()    

    model = ModelClass.load(checkpoint_name)

    
    obs, info = env.reset()

    frames = []
    done = False

    renderer = mujoco.Renderer(env.sim.model, height=368, width=640)
    
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = 1.5


    while not done:

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        renderer.update_scene(env.sim.data, camera)
        frame = renderer.render()

        frames.append(frame)
        
        done = terminated or truncated
        
    env.close()
    imageio.mimsave(
        "episode.gif",
        frames,
        fps=30
    )
    


