#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 2017/8/23-10:29
# :package: tforce.ops

import tensorflow as tf


def relu(x):
    return tf.where(x > 0., x, x * 0.)


def lrelu(x, leak=0.2):
    return tf.where(x > 0., x, x * leak)


def selu(x, alpha=1.6732632423543772848170429916717, scale=1.0507009873554804934193349852946):
    return scale * tf.where(x > 0., x, alpha * tf.exp(x) - alpha)


def dropout_relu(x, keep_prob=0.5):
    return tf.nn.dropout(x, keep_prob)


def dropout_selu(x, keep_prob=0.95, alpha=1.6732632423543772848170429916717):
    dropout = tf.nn.dropout(x, keep_prob)
    return tf.where(dropout == 0. and x != 0., dropout - alpha, x)
