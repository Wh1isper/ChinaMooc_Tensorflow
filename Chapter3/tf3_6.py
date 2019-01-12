#coding:utf-8

import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
seed = 23455

rng = np.random.RandomState(seed)

#从X这个32行2列的矩阵中抽出一行，判断两个变量之和是否小于1
X = rng.rand(32,2)
Y = [[int (x0+x1 <1)]for (x0,x1) in X]
print("X:",X)
print("Y:",Y)

#前向传播
x = tf.placeholder(tf.float32, shape=(None,2))
y_ = tf.placeholder(tf.float32, shape=(None,1))

w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#反向传播
loss = tf.reduce_mean(tf.square(y-y_))
optimizer = tf.train.GradientDescentOptimizer(0.001)
# optimizer = tf.train.MomentumOptimizer(0.001,0.9)
# optimizer = tf.train.AdamOptimizer(0.001)
train_step = optimizer.minimize(loss)

#生成会话 训练STEPS轮次
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    #未训练参数值
    print("w1:", sess.run(w1))
    print("w2:", sess.run(w2))

    #训练模型
    STEPS = 3000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32 #32代表原有数据32行 BATCH_SIZE是每次喂入数据数目
        end = start + BATCH_SIZE
        #print(start,end)
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print("After {} training step(s),loss on all data is {}".format(i,total_loss))
    print("w1:", sess.run(w1))
    print("w2:", sess.run(w2))