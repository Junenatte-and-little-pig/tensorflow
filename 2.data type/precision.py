# -*- encoding: utf-8 -*-
import numpy as np
import tensorflow as tf


def main():
    # 通常32已经够用了，如果不够就用64
    integer16 = tf.constant(1234567890123456789, dtype=tf.int16)
    integer32 = tf.constant(1234567890123456789, dtype=tf.int32)
    integer64 = tf.constant(1234567890123456789, dtype=tf.int64)
    print(integer16)
    print(integer32)
    print(integer64)

    float16 = tf.constant(np.pi, dtype=tf.float16)
    float32 = tf.constant(np.pi, dtype=tf.float32)
    float64 = tf.constant(np.pi, dtype=tf.float64)
    print(float16)
    print(float32)
    print(float64)

    # 类型转换
    # 高转低可能会导致精度丢失或者溢出
    # 布尔型和整型可以互换
    print(tf.cast(integer16,dtype=tf.int64))



if __name__ == '__main__':
    main()
