from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.models.modelv2 import ModelV2
import numpy as np

from models.impala_cnn_tf import ImpalaBase

tf1,tf,tfv = try_import_tf()


class ImpalaCNN_RND(TFModelV2):
    """
    Network from IMPALA paper implemented in ModelV2 API.

    Based on https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet_v2.py
    and https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/models.py#L28
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        args = model_config['custom_model_config']
        self.a,self.f = args['augment'],args['framestack']

        # self._framestack = args['framestack']
        obs_shape = obs_space.shape
        if args['framestack']:
            s,h,w,c = obs_space.shape
            obs_shape = (h,w,c*s)
            self.frames = tf.keras.Sequential([
                tf.keras.layers.Permute((2,3,1,4)),
                tf.keras.layers.Reshape(obs_shape)
            ])

        feature_extractor = ImpalaBase(1, name='feature_tgt', hidden_act=None)
        feature_predictor = tf.keras.Sequential([
            ImpalaBase(1, name='feature_pred'),
            tf.keras.layers.Dense(256)
        ])
        feature_extractor.trainable = False

        body = ImpalaBase(2)
        inputs = tf.keras.layers.Input(shape=obs_shape, name="observations")
        x = body(inputs)
        logits = tf.keras.layers.Dense(units=num_outputs, name="pi",)(x)
                    # kernel_initializer=tf.keras.initializers.he_normal())(x)
        value = tf.keras.layers.Dense(units=1, name="vf",)(x)
                    # kernel_initializer=tf.keras.initializers.he_normal())(x)

        tgt_features = tf.stop_gradient(feature_extractor(inputs))
        pred_features = feature_predictor(inputs)

        # self.base_model = tf.keras.Model(inputs, [logits, value])
        self.base_model = tf.keras.Model(inputs, [logits, value, tgt_features, pred_features])
        self.register_variables(self.base_model.variables)
        self.base_model.summary()
        print(len(self.base_model.trainable_variables))

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        # Transform Framestack
        if self.f:
            obs = self.frames(obs)

        # Augment inputs
        if self.a:
            obs = Augment()(obs, training=input_dict['is_training'])

        # logits, self._value  = self.base_model(obs)
        logits, self._value, tgts, preds  = self.base_model(obs, training=input_dict['is_training'])
        self._int_rew = tf.keras.losses.mean_squared_error(tgts, preds)
        return logits, state

    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        return policy_loss + self.intrinsic_rewards()

    def value_function(self):
        return tf.reshape(self._value, [-1])

    def intrinsic_rewards(self):
        return self._int_rew


# Register model in ModelCatalog
ModelCatalog.register_custom_model("impala_cnn_tf_rnd", ImpalaCNN_RND)
