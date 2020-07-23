# -*- encoding: utf-8 -*-
import tensorflow as tf


def main():
    a = tf.zeros([3, 3])
    b = tf.ones([3, 3])
    cond = tf.constant(
        [[True, False, False], [False, True, True], [False, True, True]])
    # 通过where在不同位置从两个张量中抽取数据进行采样
    print(tf.where(cond, a, b))
    # 省略采样矩阵则会返回condition张量中True的位置索引
    print(tf.where(cond))


if __name__ == '__main__':
    main()
