#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 2017/8/24-18:30
# :package: tforce.widgets.utils

import tensorflow as tf

from ...core import Widget


@Widget.from_op
def to_dense(x):
    return tf.cast(x, Widget.default.float_dtype) / 128 - 0.99609375


@Widget.from_op
def from_dense(x):
    return tf.cast(tf.clip_by_value(x, -1., 1.) * 128 + 127.5, tf.uint8)


@Widget.from_op
def to_flat(x):
    return tf.reshape(x, (tf.shape(x)[0], -1))


@Widget.from_op
def from_flat(x, height, width):
    return tf.reshape(x, (tf.shape(x)[0], height, width, -1))


@Widget.from_op
def dropout(x, keep_prob=0.5):
    return tf.nn.dropout(x, keep_prob)


@Widget.from_op
def dropout_alpha(x, keep_prob=0.95, alpha=1.6732632423543772848170429916717):
    ori = tf.nn.dropout(x, keep_prob)
    return tf.where(ori == 0. and x != 0., ori - alpha, x)
