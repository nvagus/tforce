#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 2017/8/24-18:30
# :package: tforce.widgets.utils

import tensorflow as tf

from ...core import Widget
import numpy as np


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
def randomize_shift(x, rotation=0., height_shift=0., width_shift=0.):
    theta = np.pi / 180. * tf.random_uniform((), -rotation, rotation)
    cos_theta = tf.cos(theta)
    sin_theta = tf.sin(theta)
    tx = tf.random_uniform((), -height_shift, height_shift)
    ty = tf.random_uniform((), -width_shift, width_shift)
    rotation_mat = tf.stack(
        [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
    )
    shift_mat = tf.stack(
        [[1, 0, tx], [0, 1, ty], [0, 0, 1]]
    )
    y = tf.tensordot(x, rotation_mat @ shift_mat, axes=1)
    y.set_shape(x.get_shape())
    return y
