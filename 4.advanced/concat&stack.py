# -*- encoding: utf-8 -*-
import tensorflow as tf


def main():
    a = tf.random.normal([10, 35, 4])
    b = tf.random.normal([10, 35, 4])
    # concat合并时不增加新的维度
    # axis指定合并维度，其余维度必须相同
    c = tf.concat([a, b], axis=2)
    print(c)

    # stack合并时增加新的维度，两个张量形状要一样
    # axis指定新的维度的位置，正序从0开始，倒序从-1开始
    d = tf.stack([a, b], axis=0)
    print(d)


if __name__ == '__main__':
    main()
