# -*- encoding: utf-8 -*-
from tensorflow import keras
from tensorflow.keras import layers, Sequential


class Model(keras.Model):
    # 建立网络模型
    def __init__(self):
        super(Model, self).__init__()
        self.model = Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

    def call(self, inputs, training=None, mask=None):
        x = self.model(inputs)
        return x
