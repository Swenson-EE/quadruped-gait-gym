from shared.algorithm.algorithm_types import Algorithm
from environment.base_bittle_environment import BaseBittleEnvironment, EnvironmentParameters

from typing import Type


def get_algo_model(algo: Algorithm):
    ModelClass = None
    model_parameters = {}

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

        case Algorithm.A2C:
            from stable_baselines3 import A2C
            ModelClass = A2C
        
        case _:
            ModelClass = None

    return ModelClass, model_parameters



def get_algorithm_class(algorithm: Algorithm) -> tuple[Type[BaseBittleEnvironment] | None, Type[EnvironmentParameters] | None]:
    match algorithm:
        case Algorithm.SAC | Algorithm.DDPG | Algorithm.PPO_C:
           from environment.continuous_bittle_environment import ContinuousBittleEnvironment, ContinuousEnvironmentParameters
           return ContinuousBittleEnvironment, ContinuousEnvironmentParameters
        case Algorithm.PPO_D | Algorithm.A2C:
            print(f'Discrete algorithm ({algorithm})')
            from environment.discrete_bittle_environment import DiscreteBittleEnvironment, DiscreteEnvironmentParameters
            return DiscreteBittleEnvironment, DiscreteEnvironmentParameters
        case _:
            print("Invalid algorithm:", algorithm)    

    return None


def get_algo_environment(algo: Algorithm, parameters = {}, weights = {}):
    algorithm_class, algo_env_parameters = get_algorithm_class(algo)

    if algorithm_class is not None:
        return algorithm_class(parameters=algo_env_parameters(**parameters), weights=weights)
            
    return None



def get_algo_vec_environment(algo_env_class: Type[BaseBittleEnvironment], parallel_env: int, parameters: EnvironmentParameters = EnvironmentParameters(), weights = {}):

    
    
    env: BaseBittleEnvironment = None
    
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
    
    #algo_env_class, algo_env_parameters = get_algorithm_class(algo)

    if algo_env_class is not None:
        from environment.wrappers import DeltaActionWrapper

        env = make_vec_env(
            #lambda: DeltaActionWrapper(env=algo_env_class(parameters=parameters), max_delta=45),
            lambda: algo_env_class(parameters=parameters, weights=weights),
            n_envs=parallel_env,
            vec_env_cls=SubprocVecEnv
        )

    return env


