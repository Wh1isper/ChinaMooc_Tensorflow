import tensorflow as tf
import forward
import numpy as np
import generateds
import os
import test

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.005    #0.005
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 30000
NUM_EXAMPLES = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "cifar-10_model"


def backward():
    x = tf.placeholder(tf.float32, [BATCH_SIZE, forward.IMAGE_SIZE, forward.IMAGE_SIZE, forward.NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])
    y = forward.forward(x, True, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)
    # 交叉熵
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    # 正则
    loss = cem + tf.add_n(tf.get_collection('losses'))
    # 指数下降学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, NUM_EXAMPLES / BATCH_SIZE,
                                               LEARNING_RATE_DECAY, staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 滑动平均值
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')
    # 实例化保存模型
    saver = tf.train.Saver()
    img_batch, label_batch = generateds.get_tfrecord(BATCH_SIZE, isTrain=True)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(STEPS):
            # 通过sess.run获得样本和标签
            xs, ys = sess.run([img_batch, label_batch])
            reshaped_xs = np.reshape(xs, (BATCH_SIZE, forward.IMAGE_SIZE, forward.IMAGE_SIZE, forward.NUM_CHANNELS))
            for j in range(50):
                # 对每组样本训练50次
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            if i % 10 == 0:
                print("Now Step{}".format(i))   # 提示当前步数
            if i % 1000 == 0:   # 每一千步验证一次准确率，实际上训练了1000*50=50000次
                loss_value = sess.run(loss,feed_dict={x: reshaped_xs, y_: ys})
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                print("After {} training step(s), loss on training batch is {}." .format(i, loss_value))
                test.test()
        coord.request_stop()
        coord.join(threads)

def main():
    backward()


if __name__ == '__main__':
    main()
