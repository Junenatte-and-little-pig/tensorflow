# -*- encoding: utf-8 -*-
import tensorflow as tf


def main():
    a = tf.random.normal([5, 3, 8])
    # reduce_*对张量进行数据分析
    # axis指定时给出局部值，不指定时给出全局值
    print(tf.reduce_max(a, axis=2))
    print(tf.reduce_min(a, axis=2))
    print(tf.reduce_mean(a, axis=2))
    print(tf.reduce_sum(a, axis=2))

    # argmax和argmin给出当取得最大最小值时，最大最小值所在的位置
    print(tf.argmax(a, axis=2))
    print(tf.argmin(a, axis=2))


if __name__ == '__main__':
    main()
