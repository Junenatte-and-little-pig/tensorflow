# -*- encoding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, metrics, Sequential


def main():
    # 加载MNIST数据集
    (x, y), (x_val, y_val) = datasets.mnist.load_data()
    # 转换为浮点张量，并缩放到-1~1
    x = 2 * tf.convert_to_tensor(x, dtype=tf.float32) / 255. - 1
    # 构建数据集对象
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    # 批量训练
    train_dataset = train_dataset.batch(512)

    # 网络搭建
    # 利用Sequential容器封装3个网络层，前网络层的输出默认作为下一层的输入
    # 3个非线性层的嵌套模型
    model = Sequential([
        # 隐藏层1
        layers.Dense(256, activation="relu"),
        # 隐藏层2
        layers.Dense(128, activation="relu"),
        # 输出层，输出节点数为10
        layers.Dense(10)
    ])
    # 构建网络模型，规定输入样本形状
    model.build(input_shape=(4, 28 * 28))
    # 打印网络模型信息
    model.summary()

    # 优化器
    optimizer = optimizers.SGD(lr=0.01)
    # 度量准确度
    acc = metrics.Accuracy()

    for step, (x, y) in enumerate(train_dataset):
        # 构建梯度记录环境
        with tf.GradientTape() as tape:
            # 打平数据，[b, 28, 28] => [b, 784]
            x = tf.reshape(x, (-1, 28 * 28))
            # 得到模型输出
            out = model(x)
            y_onehot = tf.one_hot(y, depth=10)
            # 计算差的平方和
            loss = tf.square(out - y_onehot)
            # 计算每个样本的平均误差
            loss = tf.reduce_sum(loss) / x.shape[0]
        # 更新准确度
        acc.update_state(tf.argmax(out, axis=1), y)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 20 == 0:
            # 输出损失并计算准确度
            print(step, 'loss:', float(loss), 'acc:', acc.result().numpy())
            # 重置准确度
            acc.reset_states()


if __name__ == '__main__':
    main()
