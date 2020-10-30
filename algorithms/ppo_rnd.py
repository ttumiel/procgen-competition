from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.agents.ppo import PPOTFPolicy, PPOTrainer
from ray.rllib.evaluation.postprocessing import compute_advantages
import numpy as np

def add_intrinsic_rews_to_batch(policy):
    return {
        SampleBatch.VF_PREDS: policy.model.value_function(),
        "int_rew": policy.model.intrinsic_rewards(),
    }

def add_intrinsic_rewards_to_gae(policy,
                        sample_batch,
                        other_agent_batches=None,
                        episode=None):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""

    completed = sample_batch[SampleBatch.DONES][-1]
    if completed:
        last_r = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append(sample_batch["state_out_{}".format(i)][-1])
        last_r = policy._value(sample_batch[SampleBatch.NEXT_OBS][-1],
                               sample_batch[SampleBatch.ACTIONS][-1],
                               sample_batch[SampleBatch.REWARDS][-1],
                               *next_state)

    # print(np.mean(sample_batch['int_rew']), np.std(sample_batch['int_rew']), np.max(sample_batch['int_rew']))
    sample_batch[SampleBatch.REWARDS] = sample_batch[SampleBatch.REWARDS] +  np.clip(sample_batch['int_rew'], 0, 2)
    batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return batch

PPOTFPolicy_RND = PPOTFPolicy.with_updates(
    name="PPOTFRND",
    postprocess_fn=add_intrinsic_rewards_to_gae,
    extra_action_fetches_fn=add_intrinsic_rews_to_batch
)

PPOTrainer_RND = PPOTrainer.with_updates(name="PPO_RND", default_policy=PPOTFPolicy_RND, get_policy_class=None)
