# -*- encoding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.saving.saved_model.load import metrics


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
    # 加载saved_model.save保存的网络
    network = tf.saved_model.load('model-savedmodel')
    # 准确率计量器
    acc_meter = metrics.CategoricalAccuracy()
    for x, y in ds_val:  # 遍历测试集
        pred = network(x)  # 前向计算
        acc_meter.update_state(y_true=y, y_pred=pred)  # 更新准确率统计
    # 打印准确率
    print("Test Accuracy:%f" % acc_meter.result())


if __name__ == '__main__':
    main()
