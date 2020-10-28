from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models import ModelCatalog
# import tensorflow_addons as tfa

tf1,tf,tfv = try_import_tf()


def conv(c, k=3, st=1, act=tf.keras.layers.ReLU, name="", gamma_init='ones'):
    return tf.keras.Sequential([
                tf.keras.layers.Conv2D(c, k, strides=st, padding='same', use_bias=False),
                # use_bias=False, kernel_initializer=tf.keras.initializers.HeNormal()),
            # tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init),
            # tfa.layers.GroupNormalization(gamma_initializer=gamma_init),
    ]+([act()] if act is not None else []))


class XResBlock(tf.keras.layers.Layer):
    def __init__(self, channels, st=1, ks=3, act=tf.keras.layers.ReLU, prefix=""):
        super().__init__(name=prefix+"_xresblock")
        self.d,self.st,self.ks,self.act = channels,st,ks,act

    def build(self, shape):
        self.cin = shape[-1]
        self.conv = tf.keras.Sequential([
            conv(self.d, st=self.st, act=self.act, name="_conv0"),
            conv(self.d, act=None, name="_conv1", gamma_init='zeros')
        ])
        short = []
        if self.st>1: short.append(tf.keras.layers.AveragePooling2D(self.st, self.st))
        if self.cin!=self.d: short.append(conv(self.d, k=1, act=None))
        self.shortcut = tf.keras.Sequential(short)

    def call(self,x, training=None):
        return self.act()(self.shortcut(x, training=training) + self.conv(x, training=training))


class XResNetBase(tf.keras.Model):
    def __init__(self, hidden_act='relu'):
        super().__init__()
        self.features = tf.keras.Sequential([
            XResBlock(channels, prefix=f"seq{i}", st=stride)
            for i,(channels,stride) in enumerate([[16,2],[32,2],[64,2],[128,2]])
        ])

        self.head = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(units=256, activation=hidden_act, name="hidden")
        ])

    def call(self, inputs, training=None):
        x = (tf.cast(inputs, tf.float32) - 128.0) / 73.9
        x = self.features(x, training=training)
        return self.head(x)


class XResNet(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        args = model_config['custom_model_config']

        inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        body = XResNetBase()
        x = body(inputs)
        logits = tf.keras.layers.Dense(units=num_outputs, name="pi",)(x)
        value = tf.keras.layers.Dense(units=1, name="vf",)(x)

        self.base_model = tf.keras.Model(inputs, [logits, value])
        self.register_variables(self.base_model.variables)
        self.base_model.summary()

    def forward(self, input_dict, state, seq_lens):
        obs = tf.cast(input_dict["obs"], tf.float32)
        logits, self._value = self.base_model(obs, training=input_dict['is_training'])
        return logits, state

    def value_function(self):
        return tf.reshape(self._value, [-1])


# Register model in ModelCatalog
ModelCatalog.register_custom_model("xresnet_tf", XResNet)
