# opt4_5.py
# coding:utf-8
# 设损失函数loss = (w+1)^2, 令w初值为10。
# 使用指数衰减的学习率，在迭代初期得到较高的下降速度

import tensorflow as tf

LEARNING_RATE_BASE = 0.1    # 最初学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减度
LEARNING_RATE_STEP = 1  # 喂入多少轮BATCH_SIZE后更新学习率，一般设为总样本数/BATCH_SIZE

# 运行了几轮BATCH_SIZE的计数器，初值为0，不可训练
global_step = tf.Variable(0, trainable=False)
# 定义指数下降学习率 staircase=True取整呈阶梯形下降，False呈平滑曲线
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,
                                           LEARNING_RATE_STEP,LEARNING_RATE_DECAY,
                                           staircase=True)
# 定义待优化参数，初始值10
w = tf.Variable(tf.constant(10, dtype=tf.float32))
# 定义损失函数
loss = tf.square(w+1)
# 定义反向传播方法 训练时记录global_step
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(loss,global_step=global_step)
# 生成会话，训练40轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        global_step_val = sess.run(global_step)
        w_val = sess.run(w)
        learning_rate_val = sess.run(learning_rate)
        loss_val = sess.run(loss)
        print("After {} steps:global_step is {}, w is {}, learning rate is {}, loss is {}".
              format(i, global_step_val, w_val, learning_rate_val, loss_val))

# 可见learning rate在不断减小