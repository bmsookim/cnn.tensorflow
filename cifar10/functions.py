import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from config import *
from tensorflow.python.framework import ops

def conv2d(x, W, strides=1):
    return tf.nn.conv2d(x, W, strides=[1,strides,strides,1], padding='SAME')

def _get_variable(name, shape, initializer, weight_decay=0.0, dtype=tf.float32, trainable=True):
    if weight_decay > 0 :
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
            shape=shape,
            initializer=initializer,
            dtype=dtype,
            regularizer=regularizer,
            collections=collections,
            trainable=trainable)

def weight_variable(name, shape, wd=CONV_WEIGHT_DECAY):
    k, c = 3, shape[-2]
    var = tf.get_variable(name,
            shape = shape,
            initializer = tf.contrib.layers.xavier_initializer_conv2d(),
            #tf.truncated_normal_initializer(stddev=np.sqrt(2.0 / (k*k*c))),
            dtype = tf.float32,
            regularizer = tf.contrib.layers.l2_regularizer(wd),
            collections = [tf.GraphKeys.VARIABLES, RESNET_VARIABLES],
            trainable = True)
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd)
        tf.add_to_collection('losses', weight_decay)
    return var

def bias_variable(shape):
    b = _get_variable('bias', shape=shape, initializer=tf.zeros_initializer)
    return b

def conv(x, n, strides=1, bias_term=True):
    W = weight_variable('weights', [3,3,channels(x),n])
    res = conv2d(x, W, strides)
    if bias_term:
        res += bias_variable([n])

    return res

def dense(x, n):
    W = _get_variable('dense', shape=[volume(x), n], initializer=tf.contrib.layers.xavier_initializer())
    b = bias_variable([n])
    return tf.matmul(flatten(x), W) + b

def activation(x):
    alphas = tf.get_variable('alpha', x.get_shape()[-1], initializer = tf.zeros_initializer, dtype=tf.float32)
    pos = tf.nn.relu(x)
    neg = alphas * (x - abs(x)) * 0.5
    return pos + neg

def max_pool(x, ksize=2, strides=2):
    return tf.nn.max_pool(x, ksize=[1,ksize,ksize,1], strides=[1,strides,strides,1], padding='SAME')

def avg_pool(x, ksize=2, strides=2):
    return tf.nn.avg_pool(x, ksize=[1,ksize,ksize,1], strides=[1,strides,strides,1], padding='SAME')

def channels(x):
    return int(x.get_shape()[-1])

def volume(x):
    return np.prod([d for d in x.get_shape()[1:].as_list()])

def flatten(x):
    return tf.reshape(x, [-1, volume(x)])

def dropout(x, keep_prob, is_train):
    def drop_prob():
        return tf.nn.dropout(x, keep_prob)
    dropout = tf.cond(is_train, drop_prob,
            lambda : tf.nn.dropout(x, 1.0))
    return dropout

def batch_norm(x, is_train):
    """Batch normalization."""
    axis = list(range(len(x.get_shape()) -1))
    params_shape = [x.get_shape()[-1]]

    beta = _get_variable('beta', params_shape, initializer=tf.zeros_initializer)
    gamma = _get_variable('gamma', params_shape, initializer=tf.ones_initializer)

    mean, variance = tf.nn.moments(x, axis, name='moments')
    moving_mean = _get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer)
    moving_variance = _get_variable('moving_variance', params_shape, initializer=tf.ones_initializer)

    moving_mean = tf.add(tf.mul(0.9, mean), tf.mul(0.1, moving_mean))
    moving_variance = tf.add(tf.mul(0.9, variance), tf.mul(0.1, moving_variance))

    tf.add_to_collection(BN_COL, moving_mean)
    tf.add_to_collection(BN_COL, moving_variance)

    mean, variance = control_flow_ops.cond(is_train,
            lambda : (mean, variance),
            lambda : (moving_mean, moving_variance))

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)

def accuracy_score(labels, logits):
    correct_prediction = tf.equal(labels, tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy
