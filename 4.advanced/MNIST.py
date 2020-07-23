# -*- encoding: utf-8 -*-
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import datasets


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)

    return x, y


def load_data(batch_size=512):
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    train_db = tf.data.Dataset.from_tensor_slices((x, y))
    train_db = train_db.shuffle(1000).batch(batch_size).map(preprocess).repeat(
        20)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.shuffle(1000).batch(batch_size).map(preprocess)

    return train_db, test_db


def main():
    train_db, test_db = load_data()
    lr = 1e-2
    accs, losses = [], []
    # 784 => 512
    w1, b1 = tf.Variable(tf.random.normal([784, 256], stddev=0.1)), tf.Variable(
        tf.zeros([256]))
    # 512 => 256
    w2, b2 = tf.Variable(tf.random.normal([256, 128], stddev=0.1)), tf.Variable(
        tf.zeros([128]))
    # 256 => 10
    w3, b3 = tf.Variable(tf.random.normal([128, 10], stddev=0.1)), tf.Variable(
        tf.zeros([10]))

    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            # layer1.
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            # layer2
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            # output
            out = h2 @ w3 + b3

            # compute loss
            loss = tf.square(y - out)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        for p, g in zip([w1, b1, w2, b2, w3, b3], grads):
            p.assign_sub(lr * g)
        if step % 100 == 0:
            print(step, "loss:", float(loss))
            losses.append(loss)
        if step % 500 == 0:
            total, total_correct = 0., 0
            for x, y in test_db:
                # layer1.
                h1 = x @ w1 + b1
                h1 = tf.nn.relu(h1)
                # layer2
                h2 = h1 @ w2 + b2
                h2 = tf.nn.relu(h2)
                # output
                out = h2 @ w3 + b3
                pred = tf.argmax(out, axis=1)
                y = tf.argmax(y, axis=1)
                correct = tf.equal(pred, y)
                total_correct += tf.reduce_sum(
                    tf.cast(correct, dtype=tf.int32)).numpy()
                total += x.shape[0]
            print(step, "evaluate acc:", total_correct / total)
            accs.append(total_correct / total)

    plt.figure()
    x = [i * 80 for i in range(len(losses))]
    plt.plot(x, losses, color='C0', marker='s', label='train')
    plt.ylabel('MSE')
    plt.xlabel('Step')
    plt.legend()
    plt.show()

    plt.figure()
    x = [i * 80 for i in range(len(accs))]
    plt.plot(x, accs, color='C1', marker='s', label='test')
    plt.ylabel('Accuracy')
    plt.xlabel('Step')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
