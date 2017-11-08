#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 2017/8/24-18:30
# :package: tforce.image

import tensorflow as tf

from ..core import Widget


def to_dense(x):
    return tf.cast(x, Widget.default.float_dtype) / 128 - 0.99609375


def from_dense(x):
    return tf.cast(tf.clip_by_value(x, -1., 1.) * 128 + 127.5, tf.uint8)


def to_flat(x):
    return tf.reshape(x, (tf.shape(x)[0], -1))


def from_flat(x, height, width):
    return tf.reshape(x, (tf.shape(x)[0], height, width, -1))


def avg_pool(x, pool_height=2, pool_width=2, stride_height=2, stride_width=2):
    return tf.nn.avg_pool(x, [1, pool_height, pool_width, 1], [1, stride_height, stride_width, 1], 'SAME')


def max_pool(x, pool_height=2, pool_width=2, stride_height=2, stride_width=2):
    return tf.nn.max_pool(x, [1, pool_height, pool_width, 1], [1, stride_height, stride_width, 1], 'SAME')
