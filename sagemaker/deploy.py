import os
import time
import yaml
import argparse

import sagemaker
from sagemaker.rl import RLEstimator, RLToolkit, RLFramework
import boto3


ROLE = ""
S3_BUCKET = ""
DOCKER_IMAGE = ""
job_name_prefix = 'sm-ray-gpu-dist-procgen'
ENVS = ["coinrun", "bigfish", "bossfight", "caveflyer",
        "chaser", "climber",  "dodgeball",
        "fruitbot", "heist", "jumper", "leaper", "maze",
        "miner", "ninja", "plunder", "starpilot"]
MACHINE_TYPE = {
    'V100': 'ml.p3.2xlarge',
    'T4': 'ml.g4dn.4xlarge'
}
METRICS =  [
    {'Name': 'training_iteration', 'Regex': 'training_iteration: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)'},
    {'Name': 'episodes_total', 'Regex': 'episodes_total: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)'},
    {'Name': 'num_steps_trained', 'Regex': 'num_steps_trained: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)'},
    {'Name': 'timesteps_total', 'Regex': 'timesteps_total: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)'},
    {'Name': 'training_iteration', 'Regex': 'training_iteration: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)'},

    {'Name': 'episode_reward_max', 'Regex': 'episode_reward_max: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)'},
    {'Name': 'episode_reward_mean', 'Regex': 'episode_reward_mean: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)'},
    {'Name': 'episode_reward_min', 'Regex': 'episode_reward_min: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)'},
]


# When using SageMaker for distributed training, you can select a GPU or CPU instance. The RLEstimator is used for training RL jobs.
#
# 1. Specify the source directory where the environment, presets and training code is uploaded.
# 2. Specify the entry point as the training code
# 3. Specify the image (CPU or GPU) to be used for the training environment.
# 4. Define the training parameters such as the instance count, job name, S3 path for output and job name.
# 5. Define the metrics definitions that you are interested in capturing in your logs. These can also be visualized in CloudWatch and SageMaker Notebooks.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='T4')
    # parser.add_argument('-f', type=str, default='tf')
    parser.add_argument('--demand', action='store_true')
    parser.add_argument('--envs', type=str, default='coinrun', help="One of {}".format(ENVS))

    args = parser.parse_args()
    print(args)

    # Setup
    # sm_session = sagemaker.session.Session()
    s3_output_path = 's3://{}/'.format(S3_BUCKET)
    # ROLE = sagemaker.get_execution_role()
    instance_type = MACHINE_TYPE[args.gpu]
    envs_to_run = args.envs.split(',') #["coinrun"]

    for env in envs_to_run:
        print("Deploying {}.".format(env))
        if not args.demand:
            job_name = 'sm-ray-dist-procgen-spot-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime()) + "-" + env
            checkpoint_s3_uri = 's3://{}/sagemaker-procgen/checkpoints/{}'.format(S3_BUCKET, job_name)
            training_params = {
                "use_spot_instances": True,
                "max_run": 3600 * 5,
                "max_wait": 7200 * 5,
                "checkpoint_s3_uri": checkpoint_s3_uri
            }
            hyperparameters = {
                "rl.training.upload_dir": checkpoint_s3_uri, # Necessary for syncing between spot instances
                "rl.training.config.env_config.env_name": env,
            }
        else:
            training_params = {"base_job_name": job_name_prefix + "-" + env}
            hyperparameters = {
                #"rl.training.upload_dir": s3_output_path + "/tensorboard_sync", # Uncomment to view tensorboard
                "rl.training.config.env_config.env_name": env,
            }

        # Defining the RLEstimator
        estimator = RLEstimator(entry_point="train-sagemaker.py",
                                source_dir='src',
                                dependencies=["src/utils", "src/common/", "src/procgen/"],
                                image_uri=DOCKER_IMAGE,
                                role=ROLE,
                                instance_type=instance_type,
                                instance_count=1,
                                output_path=s3_output_path,
                                metric_definitions=METRICS,
                                hyperparameters=hyperparameters,
                                **training_params
                            )
        if not args.demand:
            estimator.fit(job_name=job_name, wait=False)
        else:
            estimator.fit(wait=False)

        print('Deployed {} on {}'.format(env, instance_type))
        print()

if __name__ == "__main__":
    main()
