# -*- encoding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers


def main():
    x = tf.constant([2., 1., 0.1])
    # 常见网络层，有函数式接口
    out = tf.nn.softmax(x)
    print(out)
    # 更常用的，keras模块提供网络层类
    layer = layers.Softmax(axis=-1)
    out = layer(x)
    print(out)


if __name__ == '__main__':
    main()
