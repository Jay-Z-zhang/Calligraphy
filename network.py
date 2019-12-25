import tensorflow as tf
import numpy as np


class Network(object):

    def __init__(self, x, keep_prob, num_classes):
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob

        self.create()

    def create(self):
        # 卷积层1
        conv1_1 = conv(self.X, 9, 9, 128, 4, 4, padding='VALID', name='conv1_1')
        pool1 = max_pool(conv1_1, 2, 2, 2, 2, padding='SAME', name='pool1')
        # 卷积层2
        conv2_1 = conv(pool1, 4, 4, 256, 1, 1, padding='VALID', name='conv2_1')
        pool2 = max_pool(conv2_1, 2, 2, 2, 2, padding='SAME', name='pool2')
        # 卷积层3
        conv3_1 = conv(pool2, 3, 3, 512, 1, 1, padding='SAME', name='conv3_1')
        conv3_2 = conv(conv3_1, 3, 3, 512, 1, 1, padding='SAME', name='conv3_2')
        pool3 = max_pool(conv3_2, 2, 2, 2, 2, padding='SAME', name='pool3')
        # 卷积层4
        conv4_1 = conv(pool3, 3, 3, 256, 1, 1, padding='SAME', name='conv4_1')
        # pool4 = max_pool(conv4_2, 2, 2, 2, 2, padding='SAME', name='pool4')

        flattened = tf.reshape(conv4_1, [-1, 6 * 6 * 256])
        # 全链接6
        fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB)
        # 全链接7
        fc7 = fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB)
        # 全链接8
        self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, name='fc8', relu=False)


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME'):
    input_channels = int(x.get_shape()[-1])
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

        conv = convolve(x, weights)
        bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
        relu = tf.nn.relu(bias, name=scope.name)

        return relu


def fc(x, num_in, num_out, name, relu=True):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu:
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)
