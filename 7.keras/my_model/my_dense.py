# -*- encoding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers


class MyDense(layers.Layer):
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_variable('w', [inp_dim, outp_dim],
                                        trainable=True)

    def call(self, inputs, training=None):
        out = inputs @ self.kernel
        out = tf.nn.relu(out)
        return out
