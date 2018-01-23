#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/9/17-9:56 AM
# :package: tforce.widgets.utils

import functools

import tensorflow as tf

from . import image
from ...core import Widget


class AvgPool(Widget, pool_height=2, pool_width=2, stride_height=2, stride_width=2, padding='SAME'):
    def __init__(self, pool_height=None, pool_width=None, stride_height=None, stride_width=None):
        super(AvgPool, self).__init__()
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
    def instance(cls, x=None, pool_height=None, pool_width=None, stride_height=None, stride_width=None):
        return cls(pool_height, pool_width, stride_height, stride_width)(x) if x is not None else (
            lambda y: cls(pool_height, pool_width, stride_height, stride_width)(y)
        )

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


class GlobalAveragePooling(AvgPool, stride_height=1, stride_width=1, padding='VALID'):
    def __init__(self, input_height, input_width):
        super(GlobalAveragePooling, self).__init__(input_height, input_width)

    def _setup(self, x):
        x = super(GlobalAveragePooling, self)._setup(x)
        return image.to_flat(x)

    @classmethod
    def instance(cls, x=None, input_height=None, input_width=None, **kwargs):
        if x is not None:
            input_height = input_height or x.shape[1]
            input_width = input_width or x.shape[2]
            return cls(input_height, input_width)(x)
        else:
            return functools.partial(GlobalAveragePooling.instance, input_height=input_height, input_width=input_width)


class MaxPool(Widget, pool_height=2, pool_width=2, stride_height=2, stride_width=2, padding='SAME'):
    def __init__(self, pool_height=None, pool_width=None, stride_height=None, stride_width=None):
        super(MaxPool, self).__init__()
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
    def instance(cls, x=None, pool_height=None, pool_width=None, stride_height=None, stride_width=None):
        return cls(pool_height, pool_width, stride_height, stride_width)(x) if x is not None else (
            lambda y: cls(pool_height, pool_width, stride_height, stride_width)(y)
        )

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


class Dropout(Widget, keep=None):
    def __init__(self, keep_prob=0.5, alpha=0.):
        super(Dropout, self).__init__()
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
        local_dropout = tf.nn.dropout(x, self._keep_prob)
        if self._alpha != 0.:
            return tf.where(local_dropout == 0. and x != 0., local_dropout - self._alpha, x)
        else:
            return local_dropout

    @classmethod
    def instance(cls, x=None, keep_prob=0.5, alpha=0.):
        return cls(keep_prob, alpha)(x) if x is not None else (
            lambda y: cls(keep_prob, alpha)(y)
        )

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def alpha(self):
        return self._alpha


class OneHot(Widget, on_value=None, off_value=None):
    def __init__(self, labels, on_value=None, off_value=None):
        super(OneHot, self).__init__()
        self._on_value = on_value or self.default.on_value
        self._off_value = off_value or self.default.off_value
        self._labels = labels

    def _setup(self, x):
        return tf.one_hot(x, self._labels, self._on_value, self._off_value, dtype=self.default.float_dtype)

    @classmethod
    def instance(cls, x=None, labels=None, on_value=None, off_value=None):
        return cls(labels, on_value, off_value)(x) if x is not None else (
            lambda y: cls(labels, on_value, off_value)(y)
        )

    @property
    def on_value(self):
        return self._on_value

    @property
    def off_value(self):
        return self._off_value

    @property
    def labels(self):
        return self._labels


class ArgMax(Widget, axis=1):
    def __init__(self, axis=None):
        super(ArgMax, self).__init__()
        self._axis = axis or self.default.axis

    def _setup(self, x):
        return tf.argmax(x, axis=self._axis, output_type=self.default.int_dtype)

    @classmethod
    def instance(cls, x=None, axis=None):
        return cls(axis)(x) if x is not None else (lambda y: cls(axis)(y))

    @property
    def axis(self):
        return self._axis


class ArgMin(Widget, axis=1):
    def __init__(self, axis=None):
        super(ArgMin, self).__init__()
        self._axis = axis or self.default.axis

    def _setup(self, x):
        return tf.argmin(x, axis=self._axis, output_type=self.default.int_dtype)

    @classmethod
    def instance(cls, x=None, axis=None):
        return cls(axis)(x) if x is not None else (lambda y: cls(axis)(y))

    @property
    def axis(self):
        return self._axis


avg_pool = AvgPool.instance
flat_pool = GlobalAveragePooling.instance
max_pool = MaxPool.instance
dropout = Dropout.instance
one_hot = OneHot.instance
argmax = ArgMax.instance
argmin = ArgMin.instance
