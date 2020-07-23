# -*- encoding: utf-8 -*-
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def HimmelBlau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def gradient(x):
    for step in range(200):
        with tf.GradientTape() as tape:
            tape.watch([x])
            y = HimmelBlau(x)
        grads = tape.gradient(y, [x])[0]
        x -= 0.01 * grads
        if step == 199:
            print('x = {}, f(x) = {}'.format(x.numpy(), y.numpy()))


def main():
    x = np.arange(-6, 6, 0.1)
    y = np.arange(-6, 6, 0.1)
    X, Y = tf.meshgrid(x, y)
    Z = HimmelBlau([X, Y])
    fig = plt.figure('HimmelBlau')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.view_init(60, -30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

    x = tf.constant([[4., 0.], [1., 0.], [-4., 0.], [-2., 2.]])
    for i in range(4):
        gradient(x[i])


if __name__ == '__main__':
    main()
