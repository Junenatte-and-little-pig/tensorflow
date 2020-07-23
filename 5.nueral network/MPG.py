# -*- encoding: utf-8 -*-
import pandas as pd
import tensorflow as tf
from model import MPGNetwork
from tensorflow import keras
from tensorflow.keras import optimizers, losses


def norm(dataset, stats):
    return (dataset - stats.mean()) / stats.std()


def load_data():
    # 在线下载汽车效能数据集
    dataset_path = keras.utils.get_file("auto-mpg.data",
                                        "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    # 利用 pandas 读取数据集，字段有效能（公里数每加仑），气缸数，排量，马力，重量，加速度，型号年份，产地
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                              na_values="?", comment='\t',
                              sep=" ", skipinitialspace=True)
    dataset = raw_dataset.copy()
    # 查看部分数据
    # print(dataset.head())
    # 统计空白数据
    # dataset.isna().sum()
    # 删除空白数据项
    dataset = dataset.dropna()
    # 处理类别型数据，其中 origin 列代表了类别 1,2,3,分布代表产地：美国、欧洲、日本
    # 先弹出(删除并返回)origin 这一列
    origin = dataset.pop('Origin')
    # 根据 origin 列来写入新的 3 个列
    dataset['USA'] = (origin == 1) * 1.0
    dataset['Europe'] = (origin == 2) * 1.0
    dataset['Japan'] = (origin == 3) * 1.0
    # 查看新表格的后几项
    # print(dataset.tail())
    # 切分为训练集和测试集
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # 移动 MPG 油耗效能这一列为真实标签 Y
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')

    return train_dataset, train_labels, test_dataset, test_labels


def train(train_db, lr=0.001):
    model = MPGNetwork.Model()
    model.build(input_shape=(4, 9))
    # 打印网络信息
    # model.summary()
    optimizer = optimizers.RMSprop(lr)
    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            out = model(x)
            # 计算MSE
            MSE_loss = tf.reduce_mean(losses.MSE(y, out))
            # 计算MAE
            MAE_loss = tf.reduce_mean(losses.MAE(y, out))
        if step % 10 == 0:
            print(step, "MSE-loss:", float(MSE_loss), "MAE-loss:",
                  float(MAE_loss))
        grads = tape.gradient(MSE_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


def main():
    train_dataset, train_labels, test_dataset, test_labels = load_data()
    train_stats = train_dataset.describe()
    # 正则化
    norm_train_dataset = norm(train_dataset, train_stats)
    norm_test_dataset = norm(test_dataset, train_stats)
    # print(norm_train_dataset.shape, train_labels.shape)
    # print(norm_test_dataset.shape, test_labels.shape)
    # 构建 Dataset 对象
    train_db = tf.data.Dataset.from_tensor_slices(
        (norm_train_dataset.values, train_labels.values))
    # 随机打散，批量化
    train_db = train_db.shuffle(100).batch(32).repeat(100)
    train(train_db)


if __name__ == '__main__':
    main()
