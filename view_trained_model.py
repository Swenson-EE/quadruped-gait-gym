from dataclasses import dataclass, field
import time

import mujoco.viewer

from runner.training_parser import build_parser_from_dataclass
from shared.algorithm.algorithm_info import get_algo_environment, get_algo_model
from shared.algorithm.algorithm_types import Algorithm
from checkpoints.checkpoints_names import get_checkpoint, get_latest_checkpoint


@dataclass
class TrainedModel:
    algo: Algorithm = Algorithm.PPO_C
    net_arch: list[int] = field(default_factory=lambda: [64, 64, 64, 64])
    n: int = None


if __name__ == "__main__":
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
        
    env = get_algo_environment(trained_model.algo)
    if env is None:
        print("No env instantiated")
        exit()

    ModelClass = get_algo_model(trained_model.algo)
    if ModelClass is None:
        print('No model instantiated')
        exit()    

    model = ModelClass.load(checkpoint_name)

    
    with mujoco.viewer.launch_passive(env.sim.model, env.sim.data) as viewer:
        obs, info = env.reset()

        while viewer.is_running():
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            time.sleep(0.01)

            viewer.sync()
    
    


