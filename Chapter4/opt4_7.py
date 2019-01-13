# opt4_7.py
# coding:utf-8

# 0导入模块，生成模拟数据集
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 30
seed = 2
# 基于seed产生随机数
rdm = np.random.RandomState(seed)
# 随机数返回300行2列的矩阵，表示300组坐标点(x0,x1)，
X = rdm.randn(300, 2)
# 从这300行2列的而矩阵中取出一行，判断：
# 如果两个坐标平方和小于1，则给Y赋值1，否则赋值0
# 作为输入数据集的标签（模拟标注过程）
Y_ = [int(x0 * x0 + x1 * x1 < 2) for (x0, x1) in X]
# 遍历Y中元素，将1标记为red，其余标记为blue
Y_c = [['red' if y else 'blue'] for y in Y_]
# 对数据集X和标签Y进行shape整理，n行用-1表示，把X整理为n行2列，把Y整理为n行1列
X = np.vstack(X).reshape(-1, 2)
Y_ = np.vstack(Y_).reshape(-1, 1)
print(X, Y_, Y_c)
# 用plt.scatter画出数据集
# 横坐标为X第一列元素，纵坐标为X第二列元素
plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
plt.show()


# 定义神经网络的输入、参数和输出，定义前向传播过程
def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b


x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = get_weight([2, 11], 0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x, w1)+b1) # 线性整流函数（Rectified Linear Unit, ReLU）作为激活函数

w2 = get_weight([11, 1], 0.01)
b2 = get_bias([1])
y = tf.matmul(y1,w2)+b2             # 输出层不过激活函数

# 定义损失函数
loss_mes = tf.reduce_mean(tf.square(y-y_))
loss_total = loss_mes + tf.add_n(tf.get_collection('losses')) # 均方误差的损失函数+正则化w的损失

# 定义反向传播方式：不含正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mes)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x:X[start:end], y_:Y_[start:end]})
        if i % 2000 == 0:
            loss_mes_v = sess.run(loss_mes,feed_dict={x:X, y_:Y_})
            print("After {} steps, loss is {}".format(i,loss_mes_v))
    # xx在-3到3之间以步长为0.01，yy在-3到3之间以步长为0.01生成二维网格坐标
    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
    # 将xx，yy拉直，并合成一个2列的矩阵，得到一个网格坐标点的集合
    grid = np.c_[xx.ravel(),yy.ravel()]
    # 将网格坐标点喂入神经网络，获得预测结果
    probs = sess.run(y, feed_dict={x:grid})
    # 将probs的shape调整为xx的样子
    probs = probs.reshape(xx.shape)
    print("w1:", sess.run(w1))
    print("b1:", sess.run(b1))
    print("w2:", sess.run(w2))
    print("b2:", sess.run(b2))

plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
plt.contour(xx,yy,probs,levels = [.5])
plt.show()

#定义反向传播函数：包含正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize((loss_total))

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x:X[start:end], y_:Y_[start:end]})
        if i % 2000 == 0:
            loss_mes_v = sess.run(loss_mes,feed_dict={x:X, y_:Y_})
            print("After {} steps, loss is {}".format(i,loss_mes_v))
    # xx在-3到3之间以步长为0.01，yy在-3到3之间以步长为0.01生成二维网格坐标
    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
    # 将xx，yy拉直，并合成一个2列的矩阵，得到一个网格坐标点的集合
    grid = np.c_[xx.ravel(),yy.ravel()]
    # 将网格坐标点喂入神经网络，获得预测结果
    probs = sess.run(y, feed_dict={x:grid})
    # 将probs的shape调整为xx的样子
    probs = probs.reshape(xx.shape)
    print("w1:", sess.run(w1))
    print("b1:", sess.run(b1))
    print("w2:", sess.run(w2))
    print("b2:", sess.run(b2))

plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
plt.contour(xx,yy,probs,levels = [.5])
plt.show()
