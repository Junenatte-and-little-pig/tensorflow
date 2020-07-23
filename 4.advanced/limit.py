# -*- encoding: utf-8 -*-
import tensorflow as tf


def main():
    a = tf.range(9)
    # 设置张量下限
    print(tf.maximum(a, 5))
    # 设置张量上限
    print(tf.minimum(a, 5))
    # 组合设置范围
    # tf.maximum(tf.minimum(a,7),2)
    print(tf.clip_by_value(a, 2, 7))


if __name__ == '__main__':
    main()
