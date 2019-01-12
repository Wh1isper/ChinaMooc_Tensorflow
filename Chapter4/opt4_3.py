# opt4_3.py
# coding:utf-8
# 酸奶成本9元，酸奶利润1元
# 预测多了损失大，故生成的模型倾向于少预测一些
# 对比opt4_3.py

# 0导入模块，生成数据集
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455
COST = 9
PROFIT = 1

rdm = np.random.RandomState(SEED)
X = rdm.rand(32,2)
Y_ = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1,x2) in X]  #随机生成标注y=x1+x2，带±0.05噪声

# 1定义神经网络的输入、参数和输出，定义前向传播过程
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2,1], stddev=1, seed=1)) #一层神经网络
y = tf.matmul(x,w1)

# 2定义损失函数及反向传播方法
# 定义损失函数为自定义损失函数，反向传播方法为梯度下降
loss_mse = tf.reduce_mean(tf.where(tf.greater(y,y_),(y - y_)*COST,(y_ - y)*PROFIT))
optimizer = tf.train.GradientDescentOptimizer(0.001)
train_step = optimizer.minimize(loss_mse)   #用梯度下降的方法求loss_mse极小

# 3生成会话， 训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    STEPS = 20000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32  # 32代表原有数据32行 BATCH_SIZE是每次喂入数据数目
        end = start + BATCH_SIZE
        # print(start,end)
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
            print("After {} training step(s),w1 is {}".format(i, sess.run(w1)))
    print("Final w1 is:", sess.run(w1))

# 会发现结果w1约为[0.96,0.97]小于[1,1]
