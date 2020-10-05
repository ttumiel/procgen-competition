import random
from ray.tune.schedulers import PopulationBasedTraining


def explore(config):
    # ensure we run at least one sgd iter
    if config["num_sgd_iter"] < 1:
        config["num_sgd_iter"] = 1
    return config

pbt_sched = PopulationBasedTraining(
    time_attr="time_total_s",
    metric="episode_reward_mean",
    mode="max",
    perturbation_interval=120,
    resample_probability=0.25,
    # Specifies the mutations of these hyperparams
    hyperparam_mutations={
        "lr": lambda: random.uniform(1e-5, 1e-3),
        "vf_loss_coeff": lambda: random.uniform(0.1, 0.7),
        "num_sgd_iter": lambda: random.randint(1,10),
        "train_batch_size": lambda: random.randint(1024, 2**17),
    },
    custom_explore_fn=explore)
