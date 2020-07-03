# -*- encoding: utf-8 -*-
import tensorflow as tf


def main():
    # 待优化张量
    # 有name、trainable等属性
    # 便于跟踪梯度信息
    variable = tf.Variable([[1, 2], [3, 4]])
    print(variable)
    print(variable.name)
    print(variable.trainable)


if __name__ == '__main__':
    main()
