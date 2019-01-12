# opt4_4.py
# coding:utf-8
# 设损失函数 loss = (w+1)^2,
# 令w初始值为5。反向传播求解最优w，即最小loss所对应的w。

import tensorflow as tf
# 定义参数初值5
w = tf.Variable(tf.constant(5, dtype=tf.float32))
# 定义损失函数
loss = tf.square(w+1)
# 定义反向传播方法
optimizer = tf.train.GradientDescentOptimizer(0.02)
# optimizer = tf.train.GradientDescentOptimizer(1)
# optimizer = tf.train.GradientDescentOptimizer(0.0001)
# 将此处学习率改为1，w在5和-7之间跳动，不收敛。
# 将此处学习率改为0.0001，w下降太缓慢
train_step = optimizer.minimize(loss)
# 生成会话，训练40轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        print("After {} steps: w is {}, loss is {}".format(i,sess.run(w),sess.run(loss)))

# 可得到w无限趋近于-1