# Train agents using AWS sagemaker

This folder is derived from the [sagemaker starter kit](https://github.com/aws-samples/sagemaker-rl-procgen-ray). The `deploy.py` and `docker.py` files have been modified for easier use locally.

## Setup

- Make sure your boto3 location is `us-west-2` in `~/.aws/config`.
- Symlink the procgen folder inside of the src directory.
- Deploy the docker image of the environment using `python docker.py` and take note of the name of the built image.
- Paste the docker image name, your sagemaker role, and your s3 bucket into the `deploy.py` file.
- Deploy a training run using `python deploy.py`. See `-h` flag for help.
