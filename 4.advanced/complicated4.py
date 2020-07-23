# -*- encoding: utf-8 -*-
import tensorflow as tf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():
    # 线性采样，指定范围和个数
    x = tf.linspace(-8., 8., 100)
    y = tf.linspace(-8., 8., 100)
    # 生成网格数据并返回x，y
    x, y = tf.meshgrid(x, y)
    print(x, y)

    # sinc函数
    z = tf.sqrt(x ** 2 + y ** 2)
    z = tf.sin(z) / z

    # 绘制3D图
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.contour(x.numpy(), y.numpy(), z.numpy(), 50)
    plt.show()


if __name__ == '__main__':
    main()
