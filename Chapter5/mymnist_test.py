import tensorflow as tf
import mymnist_forward as forward
import mymnist_backward as backward
from tensorflow.examples.tutorials.mnist import input_data

def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, shape=(None, 784))
        y_ = tf.placeholder(tf.float32, shape=(None, 10))

        y = forward.forward(x, backward.REGULARIZER)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('\\')[-1].split('\\')[-1]
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                print("After {} training step(s),test accuracy= {}".format(global_step,accuracy_score))
            else:
                print("NO CHECKPOINT")
                return

def main():
    mnist = input_data.read_data_sets('./data/', one_hot=True)
    test(mnist)

if __name__ == '__main__':
    main()