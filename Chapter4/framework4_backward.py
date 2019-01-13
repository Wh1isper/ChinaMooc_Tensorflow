# backward.py
# coding:utf-8
import tensorflow as tf
import framework4_forward

N = 32
BATCH_SIZE = 8
STEPS = 20000
REGULARIZER = 0.01
LEARNING_RATE_BASE = 0.1    # 最初学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减度
LEARNING_RATE_STEP = N/BATCH_SIZE  # 喂入多少轮BATCH_SIZE后更新学习率，一般设为总样本数/BATCH_SIZE
MOVING_AVERAGE_DECAY = 0.99 # 滑动平均删减率

def backward():
    x = tf.placeholder(tf.float32, shape=[None,2])
    y_ = tf.placeholder(tf.float32, shape=[None,1])
    y = framework4_forward.forward(x,REGULARIZER)
    global_step = tf.Variable(0, trainable=False)
    loss_mse = tf.reduce_mean(tf.square(y - y_))
    # 也可以使用cem表示loss
    # ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))
    # cem = tf.reduce_mean(ce)

    # 正则化
    loss = loss_mse + tf.add_n(tf.get_collection('losses'))
    # loss = cem + tf.add_n(tf.get_collection('losses'))

    # 指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        LEARNING_RATE_STEP,
        LEARNING_RATE_DECAY,
        staircase=True
    )# 见opt4_5

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss, global_step=global_step)

    # 滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(STEPS):
            start = (i * BATCH_SIZE) % N
            end = start + BATCH_SIZE
            # X是输入数据，Y_是标注
            sess.run(train_step, feed_dict={x:X[start:end], y_:Y_[start:end]})
            if i% 10000==0:
                print("After {} rounds,".format(i))

if __name__ == '__main__':
    backward()