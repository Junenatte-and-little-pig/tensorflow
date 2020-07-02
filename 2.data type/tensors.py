# -*- encoding: utf-8 -*-
import tensorflow as tf


def main():
    # 0维标量
    scalar = tf.constant(1.2)
    print(scalar)
    # 1维向量
    vector = tf.constant([1, 2., 3.3])
    print(vector)
    # 2维矩阵
    matrix = tf.constant([[1., 2.2], [3.3, 4.4]])
    print(matrix)
    # n维张量
    tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(tensor)
    # 字符串类型
    # 工具函数在tf.strings模块中
    strings = tf.constant('this is a string')
    print(strings)
    # 布尔型类型
    boolean=tf.constant(True)
    print(boolean)


if __name__ == '__main__':
    main()
