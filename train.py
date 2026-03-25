

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

    if model is None:

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


    print("Start learning")

    model.learn(
        total_timesteps=training_job.total_steps,
        progress_bar=True,
    )
    
    model.save('bittle-gait')
    
    print("Saved model")