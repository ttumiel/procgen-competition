![Procgen](./docs/envs.gif)

# Procgen competition

[Procgen](https://github.com/openai/procgen) environment efficiency and generalisation [challenge](https://www.aicrowd.com/challenges/neurips-2020-procgen-competition).

My solution placed 4th overall in the generalization track and 9th in the sample efficiency track. For training logs/ideas, see [`LOGS.md`](./LOGS.md).

Read the Arxiv paper of all the top solutions [here](https://arxiv.org/abs/2103.15332).

- [Docker](#docker)
- [Sagemaker](#sagemaker)
- [Method](#method)
  - [Algorithm](#algorithm)
  - [Environment](#environment)
  - [Model](#model)
  - [Reward](#reward)
  - [Other parameters](#other-parameters)
  - [Things that didn't work](#things-that-didnt-work)
  - [Biggest improvements (descending)](#biggest-improvements-descending)
  - [Problems](#problems)
- [Results](#results)

## Docker

Install the docker image locally to test models:

- Install [nvidia docker](https://github.com/NVIDIA/nvidia-docker).
- Build the image: `docker build . -t procgen`
- Create the image: `docker create --gpus all --shm-size 10g -it -v /path/to/procgen:/home/aicrowd/ --name pr procgen bash`
- Start and attach to the image: `docker start pr && docker attach pr`

## Sagemaker

I used AWS sagemaker for training full models. See the `sagemaker` directory.

## Method

### Algorithm

I used PPO for training. After trying IMPALA and APEX, PPO outperformed both.

### Environment

I used a modified framestack wrapper to take the first and last of every 4 frames. This was so I could get more temporal information with half the amount of data and it worked quite a bit better than no framestack and the full 4 framestack.

I tried data augmentations across the framestack but this did not help PPO/IMPALA performance. It did slightly improve APEX performance but it still performed worse than PPO. `No Augs | All Augs | Flip | Rotate | Crop Resize | Translate | Pad Crop |`


### Model

I'd say my biggest improvement to performance was adjusting the model. Increasing the model width to [32,64,64] channels drastically improved performance. Pooling the final layer also helped since beforehand most of the model parameters were in the penultimate hidden layer. I tried a few other network variations (self-attention, SE-block, pretrained resnet/mobilenet, depthwise convs, deeper and wider nets), but for some strange reason the performance was always slightly worse. One of the weirdest things was when I tried to replace the maxpool layer with a stride 2 conv but this completely destroyed the training - if anyone else saw this and knows why please let me know. A good learning rate schedule also helped here.


### Reward

I tried reward shaping (`np.sign(r)*(np.sqrt(np.abs(r)+1)-1)+0.001*r`) and it seemed to help locally with PPO but on larger tests, performance didn't improve.

I tried adding an intrinsic reward signal (Random Network Distillation) to help exploration for faster training but performance remained approximately the same. However, I didn't get the original RND method working where you ignore the episode boundaries so that you get exploration across episodes, and the authors say that was important, so that might have been the issue.


### Other parameters

I ran a few grid searches and tested out a few parameter combinations, resulting in the following:

```yaml
gamma: 0.99
kl_coeff: 0.2
kl_target: 0.01
lambda: 0.9
rollout_fragment_length: 200
train_batch_size: 8192
sgd_minibatch_size: 1024
num_sgd_iter: 3
```

### Things that didn't work

- weight decay
- different network architectures
- sticky actions
- image augmentations
- intrinsic reward
- network normalisations


### Biggest improvements (descending)

- Model width and pooling
- Modified framestack
- Good hparams


### Problems

- For a while I tried to upgrade ray to the latest version but things kept breaking and my submissions all failed.
- Pytorch performance was terrible compared to tensorflow, so I couldn't use pytorch. I think this is related to the ray version.
- Took a while to get used to the sagemaker pipeline but eventually got a script working that could deploy models from my local computer.


## Results

![Results](./docs/results.png)

