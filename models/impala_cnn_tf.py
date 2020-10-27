from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models import ModelCatalog

tf1,tf,tfv = try_import_tf()


def conv_layer(depth, name, st=1):
    return tf.keras.layers.Conv2D(
        filters=depth, kernel_size=3, strides=st, padding="same", name=name,
        # kernel_initializer=tf.keras.initializers.he_normal()
    )


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, depth, prefix):
        super().__init__(name=prefix+"_resblock")
        self.d,self.p = depth,prefix
    def build(self, shape):
        assert shape[-1] == self.d
        self.conv = tf.keras.Sequential([
            tf.keras.layers.ReLU(),
            conv_layer(self.d, name=self.p + "_conv0"),
            tf.keras.layers.ReLU(),
            conv_layer(self.d, name=self.p + "_conv1")
        ])
    def call(self,x):
        return x + self.conv(x)

class ConvSequence(tf.keras.Sequential):
    def __init__(self, depth, prefix, st=1):
        super(ConvSequence, self).__init__([
            conv_layer(depth, prefix + "_conv", st=st),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
            ResBlock(depth, prefix+"_block0"),
            ResBlock(depth, prefix+"_block1")
        ])


class ImpalaBase(tf.keras.Model):
    def __init__(self, multiplier, hidden_act='relu'):
        super().__init__(name="body")
        self.m=multiplier

        self.features = tf.keras.Sequential([
            ConvSequence(channels*self.m, f"seq{i}", st=stride)
            for i,(channels,stride) in enumerate([[16,1],[32,1],[64,1]])
        ])

        self.head = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(units=256, activation=hidden_act, name="hidden")
        ])

    def call(self, inputs, training=None):
        x = (tf.cast(inputs, tf.float32) - 128.0) / 255.0
        x = self.features(x)
        return self.head(x)

class Augment(tf.keras.layers.Layer):
    # Make sure latest tf is installed.
    def __init__(self):
        super().__init__()
        self.augs = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomTranslation(0.15, 0.15, fill_mode='reflect'),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.1, fill_mode='reflect')
        ])
    def call(self, x, training=None):
        return self.augs(x, training=training)

class ImpalaCNN(TFModelV2):
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

        if self.f:
            self.frames = tf.keras.Sequential([
                tf.keras.layers.Permute((2,3,1,4)),
                tf.keras.layers.Reshape(obs_shape)
            ])

        body = ImpalaBase(1)
        inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        x = body(inputs)
        logits = tf.keras.layers.Dense(units=num_outputs, name="pi",)(x)
                    # kernel_initializer=tf.keras.initializers.he_normal())(x)
        value = tf.keras.layers.Dense(units=1, name="vf",)(x)
                    # kernel_initializer=tf.keras.initializers.he_normal())(x)
        self.base_model = tf.keras.Model(inputs, [logits, value])
        self.register_variables(self.base_model.variables)
        self.base_model.summary()

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        # Transform Framestack
        if self.f:
            obs = self.frames(obs)

        # Augment inputs
        if self.a:
            obs = Augment()(obs, training=input_dict['is_training'])

        logits, self._value = self.base_model(obs, training=input_dict['is_training'])
        return logits, state

    def value_function(self):
        return tf.reshape(self._value, [-1])


# Register model in ModelCatalog
ModelCatalog.register_custom_model("impala_cnn_tf", ImpalaCNN)
