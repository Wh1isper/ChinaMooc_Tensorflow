# coding:utf-8

import tensorflow as tf
import numpy as np
import forward
import backward
import generateds

TEST_INTERVAL_SECS = 180
TEST_NUM = 10000  # 测试样本数


def test():
    # 每次调用将输出最近一次保存结果的正确率
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [TEST_NUM, forward.IMAGE_SIZE, forward.IMAGE_SIZE, forward.NUM_CHANNELS])
        y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])
        y = forward.forward(x, False, None)

        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        # 定义测试准确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 指定从生成器中获取数据，需要sess.run后执行
        img_batch, label_batch = generateds.get_tfrecord(TEST_NUM, isTrain=False)


        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                # 读入训练好的模型
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                # -------------------------------------------------
                # 线程协调器
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 4
                # 获取样本和标签
                xs, ys = sess.run([img_batch, label_batch])
                reshaped_xs = np.reshape(xs,
                                        (TEST_NUM, forward.IMAGE_SIZE, forward.IMAGE_SIZE, forward.NUM_CHANNELS))
                # 获得测试结果测试
                accuracy_score = sess.run(accuracy, feed_dict={x: reshaped_xs, y_: ys})

                print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))

                coord.request_stop()
                coord.join(threads)
                # -------------------------------------------------
            else:
                print('No checkpoint file found')
                return


def main():
    test()


if __name__ == '__main__':
    main()
