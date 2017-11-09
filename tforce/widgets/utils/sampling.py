#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/9/17-9:56 AM
# :package: tforce.widgets.utils

import tensorflow as tf


def avg_pool(x, pool_height=2, pool_width=2, stride_height=2, stride_width=2):
    return tf.nn.avg_pool(x, [1, pool_height, pool_width, 1], [1, stride_height, stride_width, 1], 'SAME')


def max_pool(x, pool_height=2, pool_width=2, stride_height=2, stride_width=2):
    return tf.nn.max_pool(x, [1, pool_height, pool_width, 1], [1, stride_height, stride_width, 1], 'SAME')
