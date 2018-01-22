#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/8/17-10:09 AM
# :package: tforce.widgets.utils

import tensorflow as tf

from .initializers import ZerosInitializer
from .regularizers import NoRegularizer
from ...core import Widget


class Parameter(Widget):
    def __init__(self, shape, dtype, initializer=ZerosInitializer, regularizer=NoRegularizer):
        super(Parameter, self).__init__()
        self._shape = shape
        self._dtype = dtype
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
    def instance(cls, shape, dtype, initializer, regularizer):
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
    def __init__(self, shape, dtype, initializer=ZerosInitializer, regularizer=NoRegularizer):
        super(Weight, self).__init__(shape, dtype, initializer, regularizer)

    def _setup(self):
        parameter = super(Weight, self)._setup()
        tf.get_default_graph().add_to_collection(tf.GraphKeys.WEIGHTS, parameter)
        return parameter


class Filter(Parameter):
    def __init__(self, shape, dtype, initializer=ZerosInitializer, regularizer=NoRegularizer):
        super(Filter, self).__init__(shape, dtype, initializer, regularizer)

    def _setup(self):
        parameter = super(Filter, self)._setup()
        tf.get_default_graph().add_to_collection(tf.GraphKeys.WEIGHTS, parameter)
        return parameter


class Bias(Parameter):
    def __init__(self, shape, dtype, initializer=ZerosInitializer, regularizer=NoRegularizer):
        super(Bias, self).__init__(shape, dtype, initializer, regularizer)

    def _setup(self):
        parameter = super(Bias, self)._setup()
        tf.get_default_graph().add_to_collection(tf.GraphKeys.BIASES, parameter)
        return parameter
