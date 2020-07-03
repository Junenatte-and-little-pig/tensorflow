# -*- encoding: utf-8 -*-
import tensorflow as tf


def main():
    x = tf.range(96)
    print(x)
    # 张量变形
    print(tf.reshape(x, [2, 4, 4, 3]))
    print(tf.reshape(x, [2, 16, 3]))

    xx = tf.reshape(x, [2, 4, 4, 3])

    # 增加维度，值为1
    xx = tf.expand_dims(xx, axis=3)
    print(xx)

    # 删除值为1的维度
    xx = tf.squeeze(xx, axis=3)
    print(xx)

    # 交换维度
    print(tf.transpose(xx, perm=[0, 3, 1, 2]))

    # 复制，IO开销大，不推荐
    print(tf.tile(xx, multiples=[2, 1, 1, 1]))

    # 传播机制，轻量级复制，推荐
    # 在部分场景中，当+连接两个不同维度的张量时，会调用broadcast
    print(tf.broadcast_to(xx, [2, 4, 4, 3]))


if __name__ == '__main__':
    main()
