# coding:utf-8
import tensorflow as tf

IMAGE_SIZE = 28
NUM_CHANNELS = 1
CONV1_SIZE = 5
CONV1_KERNEL_NUM = 32
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64
FC_SIZE = 512
OUTPUT_NODE = 10


def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def forward(x, train, regularizer):
    # 卷积、池化，将28*28*1图片转化为7*7*64图片，减小了图片大小，增加了深度

    # 卷积1——全零填充 卷积核：（5*5*1）*32个，得到conv1：28*28*32
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    conv1 = conv2d(x, conv1_w)
    # relu激活
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    # 池化1 池化层：2*2，得到pool1：14*14*32
    pool1 = max_pool_2x2(relu1)

    # 卷积2——全零填充 卷积核：（5*5*32）*64，得到conv2：14*14*64
    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w)
    # relu激活
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    # 池化2 池化层：2*2，得到pool2：7*7*64作为输入数据
    pool2 = max_pool_2x2(relu2)
    # reshaped作为输入全连接网络的数据，由BATCH_SIZE个一维数组组成
    pool_shape = pool2.get_shape().as_list()    # [BATCH_SIZE,行数，列数，通道数（维度）]
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # 两层全连接网络
    fc1_w = get_weight([nodes, FC_SIZE], regularizer)
    fc1_b = get_bias([FC_SIZE])
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    if train: fc1 = tf.nn.dropout(fc1, 0.5) # dropout

    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b
    return y
