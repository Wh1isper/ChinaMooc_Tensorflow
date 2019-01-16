# coding:utf-8
import time
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
import mnist_generateds # 0引入生成器

TEST_INTERVAL_SECS = 5
TEST_NUM = 10000  # 1原为mnist.test.num_examples，代表测试样本数


def test():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
        y = mnist_forward.forward(x, None)

        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        # 定义测试准确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 2指定从生成器中获取数据，需要sess.run后执行
        img_batch, label_batch = mnist_generateds.get_tfrecord(TEST_NUM, isTrain=False)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 读入训练好的模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    # -------------------------------------------------
                    # 3线程协调器
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 4
                    # 获取样本和标签
                    xs, ys = sess.run([img_batch, label_batch])
                    # 获得测试结果测试
                    accuracy_score = sess.run(accuracy, feed_dict={x: xs, y_: ys})

                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))

                    coord.request_stop()
                    coord.join(threads)
                    # -------------------------------------------------
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)


def main():
    test()


if __name__ == '__main__':
    main()
