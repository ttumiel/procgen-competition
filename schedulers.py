import random
from ray.tune.schedulers import PopulationBasedTraining


def explore(config):
    # ensure we run at least one sgd iter
    # if config["num_sgd_iter"] < 1:
    #     config["num_sgd_iter"] = 1
    if config['gamma'] >= 1:
        config['gamma'] = 0.999
    if config['gamma'] <= 0:
        config['gamma'] = 0.1
    return config

def get_pbt(mutations):
    pbt_sched = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=1,
        resample_probability=0.2,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
        # {
        #     "train_batch_size": lambda: random.randint(16,512),
        #     "num_atoms": lambda: random.randint(1,100),
        #     "n_step": lambda: random.randint(1,20),
        #     "lr": lambda: random.uniform(1e-5, 1e-3),
        # },
        # {
        #     "lr": lambda: random.uniform(1e-5, 1e-3),
        #     "vf_loss_coeff": lambda: random.uniform(0.1, 0.7),
        #     "num_sgd_iter": lambda: random.randint(1,10),
        #     "train_batch_size": lambda: random.randint(1024, 2**17),
        # },

            "gamma": lambda: random.uniform(0.9, 0.999)
        },
        custom_explore_fn=explore)
    return pbt_sched
