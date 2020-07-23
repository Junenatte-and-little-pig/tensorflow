# -*- encoding: utf-8 -*-
import tensorflow as tf


def main():
    a = tf.random.normal([5, 3, 8])
    print(a)

    # 通过整型数组选择采样数据
    print(tf.gather(a, [4, 2, 6, 5, 1, 3], axis=2))
    print(tf.gather_nd(a, [[0, 0], [1, 1], [2, 2]]))

    # 通过boolean数组选择采样数据
    print(tf.boolean_mask(a, [True, False, True], axis=1))


if __name__ == '__main__':
    main()
