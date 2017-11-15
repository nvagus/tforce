#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/8/17-9:04 PM
# :package: tforce.widgets.utils

import tensorflow as tf

from ...core import Widget


class MovingAverage(Widget, name='moving_average', decay=0.99):
    def __init__(self, decay=None, initial=None, shape=None, **kwargs):
        super(MovingAverage, self).__init__(**kwargs)
        self._decay = decay or self.default.decay
        self._initial = initial
        self._shape = shape or ()

    def _build(self):
        if self._initial is not None:
            self._obj = tf.Variable(self._initial)
        else:
            self._obj = tf.Variable(tf.zeros(shape=self._shape, dtype=self.default.float_dtype))
            self._lambda = tf.Variable(tf.constant(1., dtype=self.default.float_dtype))

    def _setup(self, val, shift=None):
        new_obj = self._obj * self._decay + val * (1 - self._decay)
        if shift is not None:
            new_obj = tf.assign(self._obj, tf.cond(shift, lambda: new_obj, lambda: self._obj))
        else:
            new_obj = tf.assign(self._obj, new_obj)
        if self._initial is not None:
            return new_obj
        else:
            new_lambda = self._lambda * self._decay
            if shift is not None:
                new_lambda = tf.assign(self._lambda, tf.cond(shift, lambda: new_lambda, lambda: self._lambda))
            else:
                new_lambda = tf.assign(self._lambda, new_lambda)
            return new_obj / (1 - new_lambda)

    @staticmethod
    def new_shift():
        return tf.placeholder_with_default(tf.constant(True, dtype=tf.bool), shape=())


class BatchNorm(Widget, name='batch_norm', decay=0.999, epsilon=1e-8, shift=None):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        if BatchNorm.default.shift is None:
            BatchNorm.default.shift = MovingAverage.new_shift()

    def _build(self):
        self._pop_mean = MovingAverage(initial=0., decay=self.default.decay)
        self._pop_var = MovingAverage(initial=1., decay=self.default.decay)

    def _setup(self, x):
        batch_mean, batch_var = tf.nn.moments(x, tuple(range(len(x.get_shape()))))
        global_mean = self._pop_mean(batch_mean, shift=self.default.shift)
        global_var = self._pop_var(batch_var, shift=self.default.shift)
        return (x - global_mean) * tf.rsqrt(global_var + self.default.epsilon)

    @property
    def pop_mean(self):
        return self._pop_mean

    @property
    def pop_var(self):
        return self._pop_var


class Scale(Widget):
    default_name = 'scale'

    def __init__(self, **kwargs):
        super(Scale, self).__init__(**kwargs)

    def _build(self):
        self._mean = tf.Variable(tf.constant(0., dtype=self.default.float_dtype))
        self._var = tf.Variable(tf.constant(1., dtype=self.default.float_dtype))

    def _setup(self, x):
        return (x + self._mean) * self._var

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._var


class BatchNormWithScale(Widget):
    default_name = 'batch_norm_with_scale'

    def __init__(self, **kwargs):
        super(BatchNormWithScale, self).__init__(**kwargs)

    def _build(self):
        self._bn = BatchNorm()
        self._scale = Scale()

    def _setup(self, x):
        return self._scale(self._bn(x))

    @property
    def bn(self):
        return self._bn

    @property
    def scale(self):
        return self._scale
