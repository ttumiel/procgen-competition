# Logs

These are the thoughts and experiments I had while participating in the procgen competition.

Ideas:
- baseline ppo
- augmentations
- Agent57 (all incremental improvements on the DQN algo)
  - Add Random Network Distillation as a step to NGU (https://openai.com/blog/reinforcement-learning-with-prediction-based-rewards/)


TODO:
- [x] Train Framestack
- [x] Get sagemaker training working nicely
- [x] Use PBT to get good scheduling and hparam optimisation
- [x] Add image augmentations to sampled rollouts: https://github.com/pokaxpoka/rad_procgen
- [x] Test PPO, SAC and IMPALA
- [x] Run grid search over important params for each of the above
- [x] Add curiosity bonuses: https://docs.ray.io/en/latest/rllib-training.html#rewriting-trajectories
- [ ] Do online PBT for gamma and lr (and beta when combining curiosity loss) during evaluation runs
- [ ] Conservative Q-Learning so that you aren't over-estimating the Q value at every state. https://arxiv.org/abs/2006.04779


Trials:
- Baseline plunder (4.5)
- Framestack for plunder (Oct 04, 2020 19:05) (4.2)
- LSTM for plunder (Oct 04, 2020 20:00)

Coinrun baselines:
- Base: ~7 (5e5)
- Larger network: 0 (1e5)
- LSTM: 6 (1e5)
- Impala baseline: 4.3 (5e5)
- Impala baseline (adjusted layer): 4.9 (2e6)
- Impala change LR, gamma: 5 (8e5)

Coinrun:
- Impala (No LSTM): 8 (1.5e6)
- Impala (No LSTM): 6.9 (1e5)
- Impala (LSTM): 4.1 (1e5)
- Impala (framestack): 5.5 (1e5)

Bigfish:
- Impala (Vanilla): 1.2 (best) (5e5)
- Impala (LSTM): 1 (5e5)
- Impala (framestack): 1.2 (best) (5e5)

Plunder seems like a good baseline env:
- Contains moving objects
- Has a changing objective (shoot that ships that aren't the same as you)
- Has lots of time sensitive objectives like hitting your own ships and wasting ammo.
- Still fairly dense reward (lots of ships to hit)


Coinrun:
Algo             | Baseline
APEX             |  8.6  |
IMPALA           |  8.2  |
PPO              |  8.9  |
SAC              |  1.9  |

Plunder:
Algo             | Baseline
APEX             |  4.5  |
IMPALA           |  4.0  |
PPO              |  3.4  |
SAC              |    |

Bigfish:
Algo             | Baseline
APEX             |  1.5  |
IMPALA           |  4.8  |
PPO              |  2.5  |
SAC              |  1.1  |


APEX augmentation comparison (750 seconds / 1e6 timesteps):
         | No Augs | All Augs | Flip | Rotate | Crop Resize | Translate | Pad Crop |
Coinrun  | 6.6     | 6.5      | 7.2  | 6.5    | 6.3         | 6.8       | 6.5      |
Bigfish  | 0.9     | 0.91     | 1.25 | 1.5    | 1.0         | 1.2       | 1.0      |


Coinrun APEX Grid search:
- seems like n_steps>>0 is important
- To a lesset degree, more n_atoms helps
+--------------+----------+-------------+--------+------------------+---------+----------+
| Trial name   |   n_step |   num_atoms |   iter |   total time (s) |      ts |   reward |
|--------------+----------+-------------+--------+------------------+---------+----------|
| APEX_stacked |        1 |           1 |     19 |          633.963 | 1053952 |  2.3     |
| APEX_stacked |        3 |           1 |     21 |          638.146 | 1026816 |  4.4     |
| APEX_stacked |       20 |           1 |     19 |          621.798 | 1011840 |  5.96429 |
| APEX_stacked |        1 |           3 |     21 |          643.583 | 1035648 |  1.1     |
| APEX_stacked |        3 |           3 |     21 |          657.255 | 1034496 |  6.43617 |
| APEX_stacked |       20 |           3 |     21 |          677.538 | 1029120 |  6.80525 |
| APEX_stacked |        1 |          20 |     21 |          661.799 | 1038080 |  4.4     |
| APEX_stacked |        3 |          20 |     21 |          649.646 | 1034688 |  2.1     |
| APEX_stacked |       20 |          20 |     21 |          669.884 | 1009360 |  6.50104 |
| APEX_stacked |        1 |          51 |     21 |          682.064 | 1034240 |  2.7     |
| APEX_stacked |        3 |          51 |     21 |          669.464 | 1034240 |  0       |
| APEX_stacked |       20 |          51 |     22 |          704.835 | 1050000 |  6.38978 |
| APEX_stacked |        1 |          99 |     21 |          696.1   | 1032384 |  0.1     |
| APEX_stacked |        3 |          99 |     21 |          667.443 | 1022272 |  0       |
| APEX_stacked |       20 |          99 |     21 |          675.219 | 1018240 |  6.84211 |
+--------------+----------+-------------+--------+------------------+---------+----------+


Coinrun PPO grid search:
Also gamma = 0.99 seems better than 0.999
+------------ +--------+---------------------------+----------------------+--------+------------------+--------+----------+
| Trial name  |     lr |   rollout_fragment_length |   sgd_minibatch_size |   iter |   total time (s) |     ts |   reward |
|------------ +--------+---------------------------+----------------------+--------+------------------+--------+----------|
| PPO_stacked | 0.0005 |                       256 |                  128 |    144 |          905.319 | 884736 |  6       |
| PPO_stacked | 0.001  |                       256 |                  128 |    145 |          905.946 | 890880 |  5.2     |
| PPO_stacked | 0.0005 |                       140 |                  128 |    127 |          900.962 | 853440 |  6.60377 |
| PPO_stacked | 0.001  |                       140 |                  128 |    129 |          906.793 | 866880 |  6.28571 |
| PPO_stacked | 0.0005 |                       256 |                 1024 |    148 |          901.522 | 909312 |  8.44828 |
| PPO_stacked | 0.001  |                       256 |                 1024 |    149 |          904.929 | 915456 |  7.47664 |
| PPO_stacked | 0.0005 |                       140 |                 1024 |    137 |          901.312 | 920640 |  7.77778 |
| PPO_stacked | 0.001  |                       140 |                 1024 |    138 |          904.91  | 927360 |  7       |
+------------ +--------+---------------------------+----------------------+--------+------------------+--------+----------+

Coinrun IMPALA grid search:
+--------------- +----------------+--------------------+-----------------+--------+------------------+---------+----------+
| Trial name     |   num_sgd_iter |   train_batch_size |   vf_loss_coeff |   iter |   total time (s) |      ts |   reward |
|--------------- +----------------+--------------------+-----------------+--------+------------------+---------+----------|
| IMPALA_stacked |              1 |               1024 |             0.5 |     72 |          753.038 | 1800192 |  6.06383 |
| IMPALA_stacked |              3 |               1024 |             0.5 |     68 |          757.691 |  666624 |  5.1     |
| IMPALA_stacked |             10 |               1024 |             0.5 |     68 |          753.65  |  239616 |  4.9     |
| IMPALA_stacked |              1 |               8196 |             0.5 |     69 |          754.191 | 2045952 |  5.34653 |
| IMPALA_stacked |              3 |               8196 |             0.5 |     70 |          754.511 |  801792 |  3.6     |
| IMPALA_stacked |             10 |               8196 |             0.5 |     70 |          755.911 |  350208 |  6.2     |
| IMPALA_stacked |              1 |               1024 |             0.1 |     72 |          751.218 | 1800192 |  4.70862 |
| IMPALA_stacked |              3 |               1024 |             0.1 |     68 |          760.554 |  669696 |  4.8     |
| IMPALA_stacked |             10 |               1024 |             0.1 |     68 |          752.376 |  239616 |  2.3     |
| IMPALA_stacked |              1 |               8196 |             0.1 |     69 |          754.377 | 2045952 |  3.2     |
| IMPALA_stacked |              3 |               8196 |             0.1 |     70 |          755.807 |  801792 |  5.2     |
| IMPALA_stacked |             10 |               8196 |             0.1 |     70 |          754.123 |  350208 |  4.4     |
+--------------- +----------------+--------------------+-----------------+--------+------------------+---------+----------+




Augmentations: ~2.5e5
Algo   | Aug | Base | RewClip |
Impala | 1.5 | 5.5  | 4.4 |
APEX   | 6.7 | 6.8  | 2.3 | Seems like augmentations perform best with APEX
PPO    | 5.0 | 6.3  | 6.5 | Seems like PPO benefits most from Reward shaping


IMPALA with replay: 6.5 (more stable) 8e5 ts
no replay 1e6 - 5-6.5

Upgrade to latest Ray:
- change tensorflow import to tuple: `tf1,tf,tfv = try_import_tf()`


XResNet:
- Without Kaiming (Maybe a little better)
- Try using a stem: Seem to help a bit
- Without BN: much better! (yeah this is defs the main problem)
- Without BN, with Kaiming: Fail (? This seems rather strange?)
- Try different activation (swish): nothing much
- maxpool instead of stride
  - Maxpool in shortcut: fail
  - Maxpool instead of stride: decent (about the same maybe)
- Without stem: same
- Deeper network: much better (faster training too): try do more tests on full training
- wider network: similar (exactly what width and depth should be done using full training)
- Try separable convs: SeparableConv2D: seems promising, add depth_multiplier~=4
- Try group norm: Not too bad - give it a try `import tensorflow_addons as tfa`
- Try groupnorm and weight standardisation: WS not included in TF


Trials:
PPO: 7.5 (6M steps)
APEX: 1.2 (4.7 peak) (8M steps)
IMPALA: 1.2
impala without framestack: faster but not better (1.2)
impala with same params as last decent run: 1.2
impala without replay buffer: 1.1
PPO with old impala CNN: 7.7
Impala with old CNN: 11.4
Impala with updated but still old CNN (only normalized differently): 11.5
Impala with replay buffer: 9.7
Impala strides instead of maxpool: 1.2
Apex (maxpool): 1.4
Impala with framestack, bs=1024 (maxpool): 10.0
impala with framestack, bs=2048, lr=1.5e-4 (maxpool): 8.7
Apex smaller learning rate, more epsilon timesteps: 5 at max but 1.5 at end
// impala framestack with augmentations (flip, translate): 10.1
Apex 5 atoms, lr schedule: 2.3
// Impala with augs: 10.7. SM doesn't use augs yet
PPO with augs: 4.2
Impala with augs: 7.2
ppo no augs: 5.2
ppo reward shape: 5.8
impala lr schedule: 14.3
impala gamma=0.999: 1.7
impala plunder: 4.6
impala miner: 2.2
impala coinrun: 8.3
impala 3 workers (lr sched) bigfish: 15.2, 1h15m
impala 3 workers (lr sched) coinrun: 8.1, 1h25m
impala 1024 bs bigfish: 11.4
impala 1024 bs coinrun: 8.2
impala 1024 bs bigfish (lr sched): 17.7
impala 1024 bs coinrun (lr sched):8.7
impala with replay buffer bigfish: 17.4
impala 512 bs: 12.4
Impala reward shaping (bigfish): 12.6
apex with augs (bigfish): 4.1
apex with augs (coinrun): 7.8
impala current baseline (bigfish): 18.7
impala wider cnn (128 rollout, bigfish): 7.8
impala with shorter rollouts (128): 17.1 (but faster initial training)
impala sticky actions: 15.5
impala torch with small bs (256): failed
impala with torch cnn (reduced num envs per worker): failed
impala torch with 2 workers:failed
impala with he init: 0.9
impala 200 rollout (bigfish): 17.4
impala 200 rollout (chaser): 6 (Good!)
impala 200 rollout, .995g (bigfish): 18.4 (Larger gamma might or might not help. Use online PBT or 2 value functions)
impala 200 rollout, .995g (chaser): 2.6
impala 200 rollout sticky actions (fish): 5.8 (Sticky actions don't seem to help)
impala 200 rollout sticky actions (chaser): 1.8
impala 200 rollout framestack (fish): 11.3 (much slower training, 3h)
impala 200 rollout framestack (chaser): 4.6
impala 200 rollout 2 framestack (fish): 17.2 (2h, pretty good, might be worth evaluating)
impala 200 rollout 2 framestack (chaser): 6.6
impala framestack skip 4->2 (fish): 20.1
impala framestack skip 4->2 (chaser): 7.9
impala framestack skip 8->2 (fish): poor
impala framestack skip 8->2 (miner): 5.0 (Good, would 4->2 be the same?)
impala kaiming init (fish): poor (maybe I need to do better normalization?)
impala 4x (fish): 9.1 (6M ts, 3h, much slower, performance similar?)
impala [32 64 128] (fish): 16.6
impala 4x kaiming (fish): poor (4hr)
impala gap stack normal (fish): 15.2
ppo lr sched, 200 rollout (fish): 15.4
ppo gap stack 4->2 (miner): 5.9
ppo gap stack 4->2 (plunder): 7.7 (!!)
ppo gap stack 4->2 (chaser): 4.0
ppo mobilenet (fish): fail (lr? too few layers?)
ppo num_sgd_iter=8 gap stack (fish): fail (and v slow. consider going to 2)
pretrained frozen mobilenet (1e-3) (coinrun): fail
pretrained frozen mobilenet (5e-3) (coinrun): fail
ppo with avg pool (17.6. After only 4M steps too!!): 17.7
ppo without avg pool (6x params): 6.6
ppo avg pool (fish): 23.1
ppo avg pool gap stack (fish): 22.1
ppo avg pool (plunder): 11.6 (!!)
ppo avg pool gap stack (plunder): 5.8
small xresnet (fish): fail (?)
small xresnet lr=5e-4 (fish): fail
small xresnet lr=5e-5 (fish): fail
xresnet without kaiming init lr=1e-4 (fish): fail
ppo avg pool classes (plunder): 7.4
ppo avg pool classes (miner): 6.5
Baseline xresnet [16,2],[32,2],[64,2],[128,2] (fish): 12.1
Baseline xresnet [16,2],[32,2],[64,2] (fish): 9.9
Baseline xresnet [16,2],[32,2],[32,1],[64,2],[64,1] (bigfish): 19.9
Impala avg pool (16,32,32) (plunder): 5.5 (regression from the 11.6 run?)
Baseline xresnet [16,2],[16,1],[32,2],[32,1],[64,2],[64,1] (bigfish): 21.8
Baseline xresnet [16,2],[16,1],[32,2],[32,1],[32,1],[32,1],[32,1] (bigfish): 21.4
Wide Xresnet [32,2],[32,2],[32,1],[64,2],[64,1],[128,2] (fish): 19.6
Baseline xresnet [32,2],[32,1],[64,2],[64,1],[64,1],[64,1],[64,1] (bigfish): 25.0
Baseline xresnet [32,2],[32,1],[64,2],[64,1],[64,1],[64,1],[64,1] (plunder): 5.0
Baseline xresnet (extra stride) [32,2],[32,1],[64,2],[64,1],[64,2],[64,1],[64,1] (fish): 22.5
Impala CNN gap stack (plunder): 4.7
Impala CNN 2x width (plunder): 7.8
Impala CNN 2x width augs translate,zoom (plunder): fail
Impala CNN 2x width augs translate,zoom (fish): fail
Baseline xresnet with conv bias (plunder): 5.5
Baseline xresnet Groupnorm [32,2],[32,1],[64,2],[64,1],[64,1],[64,1],[64,1] (plunder): 5.1 (slow)
Baseline xresnet SE [32,2],[32,1],[64,2],[64,1],[64,1],[64,1],[64,1] (plunder): 4.8
Impala CNN avg pool (plunder): 5.9
Baseline xresnet SA (last 2 sa) [32,2],[32,1],[64,2],[64,1],[64,1],[64,1],[64,1] (plunder): 6.2
Baseline xresnet [32,2],[32,2],[64,2],[64,1],[64,1],[64,1],[64,1], 3 sgd iters (fish): 20.9
Xresnet one cycle sched approx (3 iters, same net as above) (fish): 21.4
Impala CNN RND small target network (episode normalized) (fish): 22.0 (Very fast learning, try plunder)
Impala CNN RND small target network (not normalized, 2-clip) (fish): 23.8
Impala CNN RND small target network (not normalized, 2-clip) (plunder): 5.6
Xresnet SE (64 blocks) (fish): 20.5
Xresnet SA (last 2 layers) (fish): 19.4
impala cnn [64,1],[64,1],[64,1],[64,1] (fish): 21.2
impala cnn 2x gap stack (fish): 20.9
conv maxpool stem xresnet (fish): fail
xresnet l2 reg (no stem) (fish): 19.2
impala_cnn 2x l2 reg (fish): 21.4
xresnet maxpool instead of stride (fish): 19.8
impala_cnn 2x l2 reg (reduced one cycle lr) (fish): 25.0
impala_cnn 2x l2 reg 0.001 (fish): 25.8
RND (coinrun): 7.1
impala_cnn 4x (fish): ~20.1 after 5M steps, 3h
2x impala 0.001 l2 reg (coinrun): ~6.9
4x impala separable conv (fish): OOM
4x (64,96,128) impala separable conv (fish): OOM
2x impala 0.001 l2 reg hidden dropout (fish): 24.8
2x impala 0.001 l2 dropout 0.3, 512 hiddens (fish): 19.7
2x impala 0.001 l2 reg 512 hiddens (fish): 25.8
Impala (32,64,128) separable conv (fish): incomplete (super slow)
2x impala l2 reg 5 sgd iters (fish): 25.0 (super slow)
2x impala gamma .996 (fish): 28.7
2x impala gamma .999 (fish): 22.9
2x impala gamma .999 (coinrun): 3 (Terrible!)
2x impala 2x framestack (fish): 25.2
2x impala RND (fish): 27.7
2x impala gamma .95 (coinrun): 7.0
2x impala gap stack (coinrun): 8.7
2x impala gap stack (fish): 26.4
2x impala .995 gamma (coinrun): 6.8
2x impala RND (coinrun): 6.9 (1h20)
Gap stack RND (fish): 24.6
Impala cnn gap stack no reg (fish): 22.8
Impala cnn gap stack no reg (plunder): 9.7
Impala cnn gap stack no reg (coinrun): 8.9 (2h)
Impala cnn 4x stack (fish): 23.9
Impala cnn 4x stack (plunder): 9.6
Impala algo (fish): 17.6

Impala gap stack .995 gamma (fish): 25.8
Impala gap stack .995 gamma (coinrun): 8.8
Impala gap stack 0.5 vf coeff (coinrun): 25.5
Impala gap stack 0.5 vf coeff (fish): 9.4

vf clip param env max, .5 vf (fish): 25.1
vf clip param env max, .5 vf (coin): 9.5
vf clip param 30, .5 vf (fish): 28.4
vf clip param 30, .5 vf (coin): 9.6
vf clip param 30, .5 vf, entropy sched (fish): 27.5
vf clip param 30, .5 vf, entropy sched (coin): 8.9



PBT meta-controller:
- need num_samples: 2
- Won't fit on GPU at the same time...



Evaluated:
- Xresnet [32,2],[32,2],[64,2],[64,1],[64,1],[64,1],[64,1]: alright but not as good as the wide impala cnn. Man, there really must be something about max pooling!?
- Wide impala cnn: Excellent!! .68
  - The only one that regressed was coinrun from almost 10 to 7. Since this game is probably the easiest (the simplest move is best) I think the network is too deep and maybe we are overfitting.
- Impala 2x better lr, 0.001 l2 reg: .66 only slightly worse but probably the same
- Impala RND: .62 a bit worse than expected. Training took too long, timing out on miner, plunder, starpilot.
- Impala gap stack: .67 better at coinrun but slightly worse at most games.
- Impala gap stack remove l2 reg, revert lr sched: .69. Good. Only performed slightly worse on miner but made up for it in coinrun.
