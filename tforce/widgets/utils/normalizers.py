#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/8/17-9:04 PM
# :package: tforce.widgets.utils

import tensorflow as tf

from ...core import Widget


class MovingAverage(Widget, decay=0.99):
    def __init__(self, decay=None, initial=None, shape=None):
        super(MovingAverage, self).__init__()
        self._decay = decay if decay is not None else self.default.decay
        self._initial = initial
        self._shape = shape or ()

    def _build(self):
        if self._initial is not None:
            self._obj = tf.Variable(tf.constant(self._initial, shape=self._shape, dtype=self.default.float_dtype))
        else:
            self._obj = tf.Variable(tf.zeros(shape=self._shape, dtype=self.default.float_dtype))
            self._lambda = tf.Variable(tf.ones((), dtype=self.default.float_dtype))

    def _setup(self, val, shift=None):
        new_obj = tf.multiply(self._obj, self._decay) + val * (1 - self._decay)
        if shift is not None:
            new_obj = tf.assign(self._obj, tf.cond(shift, lambda: new_obj, lambda: self._obj))
        else:
            new_obj = tf.assign(self._obj, new_obj)
        if self._initial is not None:
            return new_obj
        else:
            new_lambda = tf.multiply(self._lambda, self._decay)
            if shift is not None:
                new_lambda = tf.assign(self._lambda, tf.cond(shift, lambda: new_lambda, lambda: self._lambda))
            else:
                new_lambda = tf.assign(self._lambda, new_lambda)
            return new_obj / (1 - new_lambda)

    @staticmethod
    def new_shift():
        return tf.placeholder_with_default(tf.constant(True, dtype=tf.bool), shape=())


class BatchNorm(Widget, decay=0.99, epsilon=1e-3, shift=None):
    def __init__(self, axes, shape):
        super(BatchNorm, self).__init__()
        if BatchNorm.default.shift is None:
            BatchNorm.default.shift = MovingAverage.new_shift()
        self._axes = axes
        self._shape = shape

    def _build(self):
        self._pop_mean = MovingAverage(initial=0., shape=self._shape, decay=self.default.decay)
        self._pop_var = MovingAverage(initial=1., shape=self._shape, decay=self.default.decay)

    def _setup(self, x):
        batch_mean, batch_var = tf.nn.moments(x, self.axes)
        global_mean = self._pop_mean(batch_mean, shift=self.default.shift)
        global_var = self._pop_var(batch_var, shift=self.default.shift)
        train = (x - batch_mean) * tf.rsqrt(batch_var + self.default.epsilon)
        valid = (x - global_mean) * tf.rsqrt(global_var + self.default.epsilon)
        return tf.cond(self.default.shift, lambda: train, lambda: valid)

    @property
    def axes(self):
        return self._axes

    @property
    def shape(self):
        return self._shape

    @property
    def pop_mean(self):
        return self._pop_mean

    @property
    def pop_var(self):
        return self._pop_var


class Scale(Widget):
    def __init__(self, shape):
        super(Scale, self).__init__()
        self._shape = shape

    def _build(self):
        self._mean = tf.Variable(tf.zeros(shape=self._shape, dtype=self.default.float_dtype))
        self._var = tf.Variable(tf.ones(shape=self._shape, dtype=self.default.float_dtype))

    def _setup(self, x):
        return (x + self._mean) * self._var

    @property
    def shape(self):
        return self._shape

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._var


class BatchNormWithScale(Widget):
    def __init__(self, axes, shape):
        super(BatchNormWithScale, self).__init__()
        self._axes = axes
        self._shape = shape

    def _build(self):
        self._bn = BatchNorm(self._axes, self._shape)
        self._scale = Scale(self._shape)

    def _setup(self, x):
        return self._scale(self._bn(x))

    @property
    def axes(self):
        return self._axes

    @property
    def shape(self):
        return self._shape

    @property
    def bn(self):
        return self._bn

    @property
    def scale(self):
        return self._scale


@Widget.from_op
def batch_normalization(x, axes=None, shape=None):
    return BatchNormWithScale(axes or list(range(len(x.shape) - 1)), shape or (x.shape[-1],))(x)
