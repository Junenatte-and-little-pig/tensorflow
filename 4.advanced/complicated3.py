# -*- encoding: utf-8 -*-
import tensorflow as tf


def main():
    indices = tf.constant([[1], [3]])
    updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7],
                            [8, 8, 8, 8]],
                           [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3],
                            [4, 4, 4, 4]]])
    # 在空白张量中指定位置批量刷新指定数据
    print(tf.scatter_nd(indices, updates, [4, 4, 4]))


if __name__ == '__main__':
    main()
