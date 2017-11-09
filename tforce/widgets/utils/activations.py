#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 2017/8/23-10:29
# :package: tforce.ops

import tensorflow as tf

from ...core import Widget


@Widget.from_op
def relu(x):
    return tf.where(x > 0., x, x * 0.)


@Widget.from_op
def lrelu(x, leak=0.2):
    return tf.where(x > 0., x, x * leak)


@Widget.from_op
def selu(x, alpha=1.6732632423543772848170429916717, scale=1.0507009873554804934193349852946):
    return scale * tf.where(x > 0., x, alpha * tf.exp(x) - alpha)
