#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 2017/8/24-20:36
# :package: tforce.ops

import tensorflow as tf

from ...core import Widget


@Widget.from_op
def categorical_cross_entropy_loss(y_pred, y_true, with_false=True, epsilon=1e-8, sum_up=tf.reduce_mean):
    true = -y_true * tf.log(y_pred + epsilon)
    if with_false:
        false = -(1 - y_true) * tf.log(1 - y_pred + epsilon)
        cross_entropy = tf.reduce_sum(true + false, axis=1)
    else:
        cross_entropy = tf.reduce_sum(true, axis=1)
    return sum_up(cross_entropy)


@Widget.from_op
def correct_prediction(y_pred, y_true, sum_up=tf.reduce_mean):
    y_pred = tf.cast(y_pred, tf.int64)
    y_true = tf.cast(y_true, tf.int64)
    return sum_up(tf.cast(tf.equal(y_pred, y_true), Widget.default.float_dtype))
