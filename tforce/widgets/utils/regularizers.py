#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/8/17-10:20 AM
# :package: tforce.widgets.utils

import tensorflow as tf

from ...core import Widget


class Regularizer(Widget):
    def __init__(self, rate=None):
        super(Regularizer, self).__init__()
        if not hasattr(self, '_f'):
            self._f = self.default.call
        self._rate = rate if rate is not None else self.default.rate

    def _setup(self, *args, **kwargs):
        loss = self._rate * self._f(*args, **kwargs)
        tf.get_default_graph().add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, loss)
        return loss


def l1_loss(x):
    return tf.reduce_sum(tf.abs(x))


def l2_loss(x):
    return tf.reduce_sum(x ** 2) / 2


class NoRegularizer(Regularizer, rate=0., call=tf.reduce_min):
    pass


class L1Regularizer(Regularizer, rate=0.001, call=l1_loss):
    pass


class L2Regularizer(Regularizer, name='l2_regularizer', rate=0.001, call=l2_loss):
    pass
