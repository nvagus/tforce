#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/8/17-10:09 AM
# :package: tforce.widgets.utils

import tensorflow as tf

from .initializers import ZerosInitializer, OnesInitializer
from .regularizers import NoRegularizer
from ...core import Widget


class Parameter(Widget):
    def __init__(self, shape, dtype=None, initializer=ZerosInitializer, regularizer=NoRegularizer):
        super(Parameter, self).__init__()
        self._shape = shape
        self._dtype = dtype or self.default_float_dtype
        self._initializer = initializer
        self._regularizer = regularizer

    def _build(self):
        self._initializer = self._initializer()
        self._regularizer = self._regularizer()

    def _setup(self):
        parameter = tf.Variable(self._initializer(self._shape, self._dtype))
        self._regularizer(parameter)
        tf.summary.histogram(self._name, parameter)
        return parameter

    @classmethod
    def instance(cls, shape, dtype=None, initializer=None, regularizer=None):
        return cls(shape, dtype, initializer, regularizer)()

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def initializer(self):
        return self._initializer

    @property
    def regularizer(self):
        return self._regularizer


class Weight(Parameter):
    def __init__(self, shape, dtype=None, initializer=ZerosInitializer, regularizer=NoRegularizer):
        super(Weight, self).__init__(shape, dtype, initializer, regularizer)

    def _setup(self):
        parameter = super(Weight, self)._setup()
        tf.get_default_graph().add_to_collection(tf.GraphKeys.WEIGHTS, parameter)
        return parameter


class Filter(Parameter):
    def __init__(self, shape, dtype=None, initializer=ZerosInitializer, regularizer=NoRegularizer):
        super(Filter, self).__init__(shape, dtype, initializer, regularizer)

    def _setup(self):
        parameter = super(Filter, self)._setup()
        tf.get_default_graph().add_to_collection(tf.GraphKeys.WEIGHTS, parameter)
        return parameter


class Bias(Parameter):
    def __init__(self, shape, dtype=None, initializer=ZerosInitializer, regularizer=NoRegularizer):
        super(Bias, self).__init__(shape, dtype, initializer, regularizer)

    def _setup(self):
        parameter = super(Bias, self)._setup()
        tf.get_default_graph().add_to_collection(tf.GraphKeys.BIASES, parameter)
        return parameter


class Power(Parameter):
    def __init__(self, shape, dtype=None, initializer=OnesInitializer, regularizer=NoRegularizer):
        super(Power, self).__init__(shape, dtype, initializer, regularizer)

    def _setup(self):
        parameter = super(Power, self)._setup()
        tf.get_default_graph().add_to_collection(tf.GraphKeys.WEIGHTS, parameter)
        return parameter


class Layer(
    Widget,
    weight_initializer=ZerosInitializer, weight_regularizer=NoRegularizer,
    filter_initializer=ZerosInitializer, filter_regularizer=NoRegularizer,
    bias_initializer=ZerosInitializer, bias_regularizer=NoRegularizer,
    power_initializer=OnesInitializer, power_regularizer=NoRegularizer
):
    def __init__(self):
        super(Layer, self).__init__()

    def _weight_variable(self, shape):
        return Weight.instance(
            shape, self.default_float_dtype, self.default.weight_initializer, self.default.weight_regularizer
        )

    def _bias_variable(self, shape):
        return Weight.instance(
            shape, self.default_float_dtype, self.default.bias_initializer, self.default.bias_regularizer
        )

    def _filter_variable(self, shape):
        return Weight.instance(
            shape, self.default_float_dtype, self.default.filter_initializer, self.default.filter_regularizer
        )

    def _power_variable(self, shape):
        return Power.instance(
            shape, self.default_float_dtype, self.default.power_initializer, self.default.power_regularizer
        )
