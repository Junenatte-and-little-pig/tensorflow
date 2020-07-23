# -*- encoding: utf-8 -*-
import tensorflow as tf


def main():
    a = tf.random.normal([10, 35, 8])
    # split将指定维度进行分割
    # num_or_size_splits指定分割方法
    # 为数字时，等长进行分割
    b = tf.split(a, num_or_size_splits=5, axis=1)
    print(b)

    # 为数组时，按照比例进行分配
    c = tf.split(a, num_or_size_splits=[5, 10, 5, 15], axis=1)
    print(c)

    # unstack(a, axis) -> split(a, num_or_size_splits=len(a), axis)
    d = tf.unstack(a, axis=2)
    print(d)


if __name__ == '__main__':
    main()
