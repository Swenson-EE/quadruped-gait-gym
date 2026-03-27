from shared.algorithm.algorithm_types import Algorithm

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



def get_algo_environment(algo: Algorithm):
    from environment.base_bittle_environment import BaseBittleEnvironment

    env: BaseBittleEnvironment = None

    match algo:
        case Algorithm.SAC | Algorithm.DDPG | Algorithm.PPO_C:
            from environment.continuous_bittle_environment import ContinuousBittleEnvironment, ContinuousEnvironmentParameters

            print(f"Continuous algorithm ({algo})")

            # set up environment
            env = ContinuousBittleEnvironment(ContinuousEnvironmentParameters())
            
        case Algorithm.QLEARNING | Algorithm.PPO_D:
            

            print(f'Discrete algorithm ({algo})')


        case _:
            print("Invalid algorithm:", algo)    
            
    return env



def get_algo_vec_environment(algo: Algorithm, parallel_env: int):
    from environment.base_bittle_environment import BaseBittleEnvironment
    
    env: BaseBittleEnvironment = None
    
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    match algo:
        case Algorithm.SAC | Algorithm.DDPG | Algorithm.PPO_C:
            from environment.continuous_bittle_environment import ContinuousBittleEnvironment, ContinuousEnvironmentParameters

            print(f"Continuous algorithm ({algo})")

            # set up environment
            env = make_vec_env(
                lambda: ContinuousBittleEnvironment(ContinuousEnvironmentParameters()),
                n_envs=parallel_env,
                vec_env_cls=SubprocVecEnv
            )
            
        case Algorithm.QLEARNING | Algorithm.PPO_D:
            

            print(f'Discrete algorithm ({algo})')


        case _:
            print("Invalid algorithm:", algo)    
            
    return env
