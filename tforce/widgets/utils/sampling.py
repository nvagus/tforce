#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/9/17-9:56 AM
# :package: tforce.widgets.utils

import tensorflow as tf

from ...core import Widget


class AvgPool(Widget, name='avg_pool', pool_height=2, pool_width=2, stride_height=2, stride_width=2, padding='SAME'):
    def __init__(self, pool_height=None, pool_width=None, stride_height=None, stride_width=None, **kwargs):
        super(AvgPool, self).__init__(**kwargs)
        self._pool_height = pool_height or self.default.pool_height
        self._pool_width = pool_width or self.default.pool_width
        self._stride_height = stride_height or self.default.stride_height
        self._stride_width = stride_width or self.default.stride_width

    def _setup(self, x):
        return tf.nn.avg_pool(
            x,
            [1, self._pool_height, self._pool_width, 1],
            [1, self._stride_height, self._stride_width, 1],
            self.default.padding
        )

    @classmethod
    def instance(cls, x, pool_height=None, pool_width=None, stride_height=None, stride_width=None):
        return AvgPool(pool_height, pool_width, stride_height, stride_width)(x)

    @property
    def pool_height(self):
        return self._pool_height

    @property
    def pool_width(self):
        return self._pool_width

    @property
    def stride_height(self):
        return self._stride_height

    @property
    def stride_width(self):
        return self._stride_width


avg_pool = AvgPool.instance


class MaxPool(Widget, name='max_pool', pool_height=2, pool_width=2, stride_height=2, stride_width=2, padding='SAME'):
    def __init__(self, pool_height=None, pool_width=None, stride_height=None, stride_width=None, **kwargs):
        super(MaxPool, self).__init__(**kwargs)
        self._pool_height = pool_height or self.default.pool_height
        self._pool_width = pool_width or self.default.pool_width
        self._stride_height = stride_height or self.default.stride_height
        self._stride_width = stride_width or self.default.stride_width

    def _setup(self, x):
        return tf.nn.max_pool(
            x,
            [1, self._pool_height, self._pool_width, 1],
            [1, self._stride_height, self._stride_width, 1],
            self.default.padding
        )

    @classmethod
    def instance(cls, x, pool_height=None, pool_width=None, stride_height=None, stride_width=None):
        return AvgPool(pool_height, pool_width, stride_height, stride_width)(x)

    @property
    def pool_height(self):
        return self._pool_height

    @property
    def pool_width(self):
        return self._pool_width

    @property
    def stride_height(self):
        return self._stride_height

    @property
    def stride_width(self):
        return self._stride_width


max_pool = MaxPool.instance


class Dropout(Widget, name='dropout', keep=None):
    def __init__(self, keep_prob=0.5, alpha=0., **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self._keep_prob = keep_prob
        self._alpha = alpha

    def _build(self):
        if self.default.keep is None:
            self.default.keep = tf.placeholder_with_default(False, ())
        self._keep_prob = tf.cond(
            self.default.keep,
            lambda: tf.constant(1., dtype=self.default.float_dtype),
            lambda: tf.constant(self._keep_prob, dtype=self.default.float_dtype)
        )

    def _setup(self, x):
        dropout = tf.nn.dropout(x, self._keep_prob)
        if self._alpha != 0.:
            return tf.where(dropout == 0. and x != 0., dropout - self._alpha, x)
        else:
            return dropout

    @classmethod
    def instance(cls, x, keep_prob=0.5, alpha=0.):
        return cls(keep_prob, alpha)(x)

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def alpha(self):
        return self._alpha


dropout = Dropout.instance


class OneHot(Widget, name='one_hot', on_value=None, off_value=None):
    def __init__(self, labels, on_value=None, off_value=None, **kwargs):
        super(OneHot, self).__init__(**kwargs)
        self._on_value = on_value or self.default.on_value
        self._off_value = off_value or self.default.off_value
        self._labels = labels

    def _setup(self, x):
        return tf.one_hot(x, self._labels, self._on_value, self._off_value, dtype=self.default.float_dtype)

    @classmethod
    def instance(cls, x, labels, on_value=None, off_value=None):
        return cls(labels, on_value, off_value)(x)

    @property
    def on_value(self):
        return self._on_value

    @property
    def off_value(self):
        return self._off_value

    @property
    def labels(self):
        return self._labels


one_hot = OneHot.instance


class ArgMax(Widget, name='arg_max', axis=1):
    def __init__(self, axis=None, **kwargs):
        super(ArgMax, self).__init__(**kwargs)
        self._axis = axis or self.default.axis

    def _setup(self, x):
        return tf.argmax(x, axis=self._axis, output_type=self.default.int_dtype)

    @classmethod
    def instance(cls, x, axis=None):
        return cls(axis)(x)

    @property
    def axis(self):
        return self._axis


argmax = ArgMax.instance


class ArgMin(Widget, name='arg_min', axis=1):
    def __init__(self, axis=None, **kwargs):
        super(ArgMin, self).__init__(**kwargs)
        self._axis = axis or self.default.axis

    def _setup(self, x):
        return tf.argmin(x, axis=self._axis, output_type=self.default.int_dtype)

    @classmethod
    def instance(cls, x, axis=None):
        return cls(axis)(x)

    @property
    def axis(self):
        return self._axis


argmin = ArgMin.instance
