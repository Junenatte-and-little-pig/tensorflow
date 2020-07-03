# -*- encoding: utf-8 -*-
import tensorflow as tf


def main():
    # 可以通过python list或者numpy array对象进行创建
    # 通过zeros、ones创建全0/全1张量，需要指定张量维度
    # 通过zeros_like、ones_like创建全0/全1张量，通过指定张量来指定张量维度
    zeros = tf.zeros([3, 2])
    print(zeros)
    ones = tf.ones([2, 3])
    print(ones)
    zeros_like = tf.zeros_like(zeros)
    print(zeros_like)
    ones_like = tf.ones_like(ones)
    print(ones_like)

    # 自定义数值填充张量
    fill = tf.fill([2, 2], -1)
    print(fill)

    # 创建符合正态分布的张量
    normal = tf.random.normal([3, 3], mean=1.0, stddev=2.0)
    print(normal)

    # 创建符合均匀分布的张量
    uniform = tf.random.uniform([3, 3], maxval=10)
    print(uniform)

    # 创建指定范围序列，含前不含后
    range = tf.range(0, 10, delta=2)
    print(range)


if __name__ == '__main__':
    main()
