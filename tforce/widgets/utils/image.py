#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 2017/8/24-18:30
# :package: tforce.widgets.utils

import numpy as np
import tensorflow as tf

from ...core import Widget


@Widget.from_op
def to_dense(x, mean=127.5, std=128):
    return (tf.cast(x, Widget.default.float_dtype) - mean) / std


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
def randomize_shift(x, rotation=0., height_shift=0., width_shift=0., switch=None):
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
    if switch is not None:
        y = tf.cond(switch, lambda: y, lambda: x)
    y.set_shape(x.get_shape())
    return y


@Widget.from_op
def randomize_crop(x, pad=4, value=0, switch=None):
    y = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], constant_values=value)
    y = tf.map_fn(lambda z: tf.random_crop(z, x.get_shape()[1:]), y)
    if switch is not None:
        y = tf.cond(switch, lambda: y, lambda: x)
    y.set_shape(x.get_shape())
    return y


@Widget.from_op
def randomize_flip(x, horizontal=0., vertical=0., switch=None):
    ch = tf.random_uniform((), 0., 1., Widget.default.float_dtype) < horizontal
    cv = tf.random_uniform((), 0., 1., Widget.default.float_dtype) < vertical
    if switch is not None:
        ch = tf.logical_and(switch, ch)
        cv = tf.logical_and(switch, cv)
    y = tf.cond(ch, lambda: tf.map_fn(tf.image.flip_left_right, x), lambda: x)
    y = tf.cond(cv, lambda: tf.map_fn(tf.image.flip_left_right, y), lambda: y)
    y.set_shape(x.get_shape())
    return y
