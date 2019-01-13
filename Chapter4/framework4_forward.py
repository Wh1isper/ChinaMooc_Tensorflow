# forward.py
# coding:utf-8
import tensorflow as tf

def forward(x,regularizer):
    w = get_weight([2,1],regularizer)
    b = get_bias([1])
    y = tf.matmul(x,w)+b
    return y

def get_weight(shape,regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    return b

