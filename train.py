

if __name__ == "__main__":
    from algorithms.algorithm_types import Algorithm
    from runner.training_job import TrainingJob, training_job_parser

    args = training_job_parser.parse_args()
    training_job = TrainingJob(**vars(args))

    print('Job:', training_job)

    from algorithms.algorithm_info import get_algo_vec_environment, get_algo_model
    env = get_algo_vec_environment(training_job.algo, training_job.parallel_env)
    if env is None:
        print("No env instantiated")
        exit()
    
    print('Created env')
    
    
    ModelClass = get_algo_model(training_job.algo)

    if ModelClass is None:
        print('No model instantiated')
        exit()

    
    model = None
    name = f'{training_job.algo}'      

    
    from checkpoints.checkpoints_names import get_latest_checkpoint

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

        custom_architecture = dict(net_arch=training_job.net_arch)

        model = ModelClass(
            'MultiInputPolicy',
            env,
            policy_kwargs=custom_architecture, # neural network

            learning_rate=training_job.lr,
            gamma=training_job.discount_factor,
            n_steps=int(training_job.batch_steps / training_job.parallel_env),        

            seed=training_job.seed,
            device=training_job.device,
            verbose=training_job.verbose
        )


    

    from checkpoints.checkpoints_names import next_checkpoint
    next_checkpoint_save, trained_name = next_checkpoint(
        algo=training_job.algo,
        # Any additional metadata for checkpoint naming
        layers=training_job.net_arch
    )

    print(f"Start learning {trained_name}")

    from loggers.multi_logger_callback import MultiLoggerCallback

    logger_callback = MultiLoggerCallback(
        name=trained_name,
        algo=training_job.algo,
        loggers=[

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