from shared.algorithm.algorithm_types import Algorithm
from environment.base_bittle_environment import BaseBittleEnvironment, EnvironmentParameters

from typing import Type


def get_algo_model(algo: Algorithm):
    ModelClass = None

    match algo:
        case Algorithm.SAC:
            from stable_baselines3 import SAC
            ModelClass = SAC
            
        case Algorithm.DDPG:
            from stable_baselines3 import DDPG
            ModelClass = DDPG

        case Algorithm.PPO_D | Algorithm.PPO_C:
            from stable_baselines3 import PPO
            ModelClass = PPO

        case Algorithm.QLEARNING:
            from stable_baselines3 import DQN
            ModelClass = DQN
        
        case _:
            ModelClass = None

    return ModelClass



def get_algorithm_class(algorithm: Algorithm) -> Type[BaseBittleEnvironment] | None:
    match algorithm:
        case Algorithm.SAC | Algorithm.DDPG | Algorithm.PPO_C:
           from environment.continuous_bittle_environment import ContinuousBittleEnvironment
           return ContinuousBittleEnvironment
        case Algorithm.QLEARNING | Algorithm.PPO_D:
            print(f'Discrete algorithm ({algorithm})')
        case _:
            print("Invalid algorithm:", algorithm)    

    return None


def get_algo_environment(algo: Algorithm, parameters: EnvironmentParameters = EnvironmentParameters()):
    algorithm_class: BaseBittleEnvironment = get_algorithm_class(algo)

    if algorithm_class is not None:
        return algorithm_class(parameters=parameters)
            
    return None



def get_algo_vec_environment(algo: Algorithm, parallel_env: int, parameters: EnvironmentParameters = EnvironmentParameters()):

    
    env: BaseBittleEnvironment = None
    
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
    
    algo_env_class = get_algorithm_class(algo)

    if algo_env_class is not None:
        from environment.wrappers import DeltaActionWrapper

        env = make_vec_env(
            lambda: DeltaActionWrapper(env=algo_env_class(parameters=parameters), max_delta=15),
            n_envs=parallel_env,
            vec_env_cls=SubprocVecEnv
        )

    return env
