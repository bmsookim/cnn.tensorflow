import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from config import *

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

def weight_variable(name, shape, wd=0.0005):
    k, c = 3, shape[-2]
    # var = tf.Variable(tf.truncated_normal(shape, stddev=np.sqrt(2.0 / (k*k*c))))
    var = tf.get_variable(name,
            shape = shape,
            initializer = tf.truncated_normal_initializer(stddev=np.sqrt(2.0 / (k*k*c))),
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

def conv(name, x, n, strides=1, bias_term=True):
    with tf.variable_scope(name) as scope :
        W = weight_variable('weights', [3,3,channels(x),n])
        res = conv2d(x, W, strides)
        if bias_term:
            res += bias_variable([n])
    return res

def dense(name, x, n):
    with tf.variable_scope(name) as scope:
        W, b = weight_variable('dense', [volume(x), n]), bias_variable([n])
    return tf.matmul(flatten(x), W) + b

def activation(x):
    return tf.nn.relu(x)

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

def _batch_norm(name, x, is_train):
    with tf.variable_scope(name) as scope:
        axis = list(range(len(x.get_shape()) -1))
        params_shape = [x.get_shape()[-1]]

        beta = tf.get_variable('beta', params_shape, tf.float64, initializer=tf.zeros_initializer)
        gamma = tf.get_variable('gamma', params_shape, tf.float64, initializer=tf.ones_initializer)

        mean, variance = tf.nn.moments(x, axis)
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-3)

def batch_norm(name, x, is_train):
    """Batch normalization."""
    with tf.variable_scope(name) as scope:
        axis = list(range(len(x.get_shape()) -1))
        params_shape = [x.get_shape()[-1]]

        beta = _get_variable('beta', params_shape, initializer=tf.zeros_initializer)
        gamma = _get_variable('gamma', params_shape, initializer=tf.ones_initializer)

        moving_mean = _get_variable('moving_mean', params_shape, tf.zeros_initializer, trainable=False)
        moving_variance = _get_variable('moving_variance', params_shape, tf.ones_initializer, trainable=False)

        mean, variance = tf.nn.moments(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
        update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)

        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

        mean, variance = control_flow_ops.cond(is_train,
                lambda : (mean, variance),
                lambda : (moving_mean, moving_variance))

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)

def accuracy_score(labels, logits):
    correct_prediction = tf.equal(labels, tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy
