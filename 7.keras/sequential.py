# -*- encoding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers, Sequential


def main():
    # 创建网络容器
    network = Sequential([
        layers.Dense(3),
        layers.ReLU(),
        layers.Dense(2),
        layers.ReLU()
    ])
    # 可以通过add添加网络层
    network.add(layers.Dense(1))
    # 创建完成需要进行build
    network.build(input_shape=(4, 4))
    # 打印网络信息
    print(network.summary())
    x = tf.random.normal([4, 4])
    # 调用call方法获得结果
    out = network(x)
    print(out)


if __name__ == '__main__':
    main()
