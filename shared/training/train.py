from environment.base_bittle_environment import EnvironmentParameters
from loggers.components_logger import ComponentsLogger
from runner.training_job import TrainingJob, training_job_parser

from shared.training.training_status import TrainingStatus

import numpy as np


def train(training_job: TrainingJob) -> tuple[TrainingStatus, str]:
    try:
        np.set_printoptions(linewidth=150)

        print("-" * 30)
        print('[Training]\n')
        print(training_job)
        print("-" * 30)

        from shared.algorithm.algorithm_info import get_algo_vec_environment, get_algo_model
        env = get_algo_vec_environment(training_job.algo, training_job.parallel_env, parameters=EnvironmentParameters(
            total_length=training_job.total_steps
        ))
        if env is None:
            return (TrainingStatus.NO_ENV, "No environment instantiated")
        
        print('Created environment')
        
        
        ModelClass, model_parameters = get_algo_model(training_job.algo)

        if ModelClass is None:
            return (TrainingStatus.NO_MODEL, "No model instantiated")

            
        model = None
        
        from shared.checkpoints.checkpoints_names import get_latest_checkpoint

        last_checkpoint = get_latest_checkpoint(
            algo=training_job.algo,
            # Any additional metadata for checkpoint naming
            layers=training_job.net_arch
        )

        if last_checkpoint:
            print(f"Found previous checkpoint ({last_checkpoint})")

            model = ModelClass.load(last_checkpoint, env=env)


        if model is None:
            print(f"Creating new checkpoint (algo={training_job.algo}, layers=({training_job.net_arch}))")
            
            model_parameters.setdefault('policy_kwargs', {})['net_arch'] = training_job.net_arch # neural network

            model = ModelClass(
                'MultiInputPolicy',
                env,
                **model_parameters,

                learning_rate=training_job.lr,
                gamma=training_job.discount_factor,
                        
                n_steps=int((training_job.batch_steps * 8) / training_job.parallel_env),

                seed=training_job.seed,
                device=training_job.device,
                verbose=training_job.verbose
            )

            print('net_arch:', model.policy_kwargs["net_arch"])


        

        from shared.checkpoints.checkpoints_names import next_checkpoint
        next_checkpoint_save, trained_name = next_checkpoint(
            algo=training_job.algo,
            # Any additional metadata for checkpoint naming
            layers=training_job.net_arch
        )

        print(f"Start learning {trained_name}")

        from loggers.multi_logger_callback import MultiLoggerCallback
        from loggers.reward_logger import RewardLogger
        from loggers.observation_logger import ObservationLogger

        logger_callback = MultiLoggerCallback(
            name=trained_name,
            algo=training_job.algo,
            loggers=[
                RewardLogger(
                    log_frequency=training_job.recording_frequency
                ),
                ObservationLogger(
                    log_frequency=training_job.recording_frequency
                ),
                ComponentsLogger(
                    log_frequency=training_job.recording_frequency
                )
            ],
            verbose=training_job.verbose
        )


        model.learn(
            total_timesteps=training_job.total_steps,
            progress_bar=True,
            callback=logger_callback,
        )
        
        model.save(next_checkpoint_save)
        
        print(f"Saved model {next_checkpoint_save}")
    except Exception as e:
        print('Error:', e)
        return (TrainingStatus.ERROR, e)
    
    return (TrainingStatus.SUCCESS, 'success')
    


if __name__ == "__main__":
    args = training_job_parser.parse_args()
    training_job = TrainingJob(**vars(args))

    result = train(training_job=training_job)
    print('Training result')
    print(result[1])
