# -*- encoding: utf-8 -*-
import tensorflow as tf


def main():
    a = tf.random.uniform([100], dtype=tf.int64, maxval=10)
    b = tf.random.uniform([100], dtype=tf.int64, maxval=10)

    # 通过一系列比较函数，比较两个同形状的张量，返回相同形状的张量
    # 结果中每个元素代表对应位置上元素比较的结果
    # tf.equal = tf.math.equal
    print(tf.equal(a, b))
    print(tf.less(a, b))
    print(tf.greater(a, b))


if __name__ == '__main__':
    main()
