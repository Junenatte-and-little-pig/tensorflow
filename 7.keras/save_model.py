# -*- encoding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, optimizers, losses


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


def main():
    (x, y), (x_val, y_val) = keras.datasets.mnist.load_data()
    print(x.shape, y.shape)
    print(x_val.shape, y_val.shape)
    batchsz = 128
    db = tf.data.Dataset.from_tensor_slices((x, y))
    db = db.map(preprocess).shuffle(60000).batch(batchsz)
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    ds_val = ds_val.map(preprocess).batch(batchsz)
    network = Sequential([
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(10),
    ])
    # 模型装配，指定优化器、损失函数、度量标准
    network.compile(optimizer=optimizers.Adam(lr=0.01),
                    loss=losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    network.build(input_shape=(4, 28 * 28))
    history = network.fit(db, epochs=5, validation_data=ds_val,
                          validation_freq=2)
    print(history.history)
    # save_weights保存网络的参数
    network.save_weights('weights.ckpt')
    # save保存网络结构及参数
    network.save('model.h5')
    # saved_model将模型及参数保存至指定文件夹
    tf.saved_model.save(network, 'model-savedmodel')
    network.summary()


if __name__ == '__main__':
    main()
