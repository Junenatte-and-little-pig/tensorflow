# -*- encoding: utf-8 -*-
import tensorflow as tf


def main():
    a = tf.random.normal([5, 3, 8])
    # pad对张量每一维度指定二元数组，将对应
    print(a)
    print(tf.pad(a, [[0, 2], [2, 0], [1, 1]]))


if __name__ == '__main__':
    main()
