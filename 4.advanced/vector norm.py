# -*- encoding: utf-8 -*-
import numpy as np
import tensorflow as tf


def main():
    a = tf.random.normal([5, 3, 8])
    # 向量范数，计算向量的长度，并且可以运用到矩阵乃至张量上去
    # L1范数：所有元素绝对值之和
    print(tf.norm(a, ord=1))
    # L2范数：所有元素平方和开根号
    print(tf.norm(a, ord=2))
    # 无限范数：所有元素绝对值的最大值
    print(tf.norm(a, ord=np.inf))


if __name__ == '__main__':
    main()
