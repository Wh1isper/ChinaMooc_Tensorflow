import os
import tensorflow as tf
import mymnist_forward as forward
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./data/', one_hot=True)
STEPS = 5000
BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.999
REGULARIZER = 0.01
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = r'.\myminist'
MODEL_NAME = 'myminist'


def backward():
    x = tf.placeholder(tf.float32, shape=(None, 784))
    y_ = tf.placeholder(tf.float32, shape=(None, 10))

    y = forward.forward(x, REGULARIZER)

    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        55000 / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )

    #loss_mse = tf.reduce_mean(tf.square(y-y_))
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss_total = cem + tf.add_n(tf.get_collection('losses'))

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss_total,global_step=global_step)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(STEPS):
            X, Y_ = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={x: X, y_:Y_})
            if i % 500 == 0:
                loss_v = sess.run(loss_total, feed_dict={x:X,y_:Y_})
                print("After {} steps, loss is {}".format(i,loss_v))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print("accuracy: ",sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("finally accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
    backward()