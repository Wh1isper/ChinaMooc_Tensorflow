# opt4_6.py
# coding:utf-8

import tensorflow as tf

# 1.定义变量及滑动平均类
# 定义一个32位浮点变量， 初始值为0，0
# 此篇代码的目的就是不断更新优化w1参数，利用滑动平均做w1的影子
w1 = tf.Variable(0, dtype=tf.float32)
# 定义num_updates（NN迭代轮数），初始值为0，不可被训练
global_step = tf.Variable(0, trainable=False)
# 实例化滑动平均类，删减率为0.99，当前轮次为global_step
MOVING_AVERAGE_DECAY = 0.99
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
# ema.apply的括号里是更新列表，每次运行sess.run(ema_op)时，对更新列表中的元素求滑动平均值
# 实际应用中使用tf.trainable_variables()自动将所有待训练参数汇总为列表
# ema_op = ema.apply([w1])
ema_op = ema.apply(tf.trainable_variables())

# 2.查看不同迭代中变量取值的变化
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 打印当前参数w1和w1的滑动平均值
    print( sess.run([w1,ema.average(w1)]) )

    # 参数w1的值赋为1
    sess.run(tf.assign(w1,1))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    # 更新step和w1的值，模拟出100轮迭代后，参数w1变为10
    sess.run(tf.assign(global_step,100))
    sess.run(tf.assign(w1,10))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    # 每次sess.run都会更新一次w1的滑动平均值
    # 多写几个看看滑动平均是否“如影随形”
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

# 更改MOVING_AVERAGE_DECAY 为 0.1，看影子追随速度
