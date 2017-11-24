#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/8/17-4:49 PM
# :package: tforce.widgets

import tensorflow as tf

from .utils import HeUniformInitializer, ZerosInitializer
from .utils import L2Regularizer, NoRegularizer
from .utils import Weight, Bias
from ..core import Widget, DeepWidget
from .utils import BatchNormWithScale


class Linear(
    Widget, name='linear',
    weight_initializer=HeUniformInitializer, weight_regularizer=L2Regularizer,
    bias_initializer=ZerosInitializer, bias_regularizer=NoRegularizer
):
    def __init__(self, input_depth, output_depth, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self._input_depth = input_depth
        self._output_depth = output_depth

    def _build(self):
        self._weight = Weight.instance(
            shape=(self._input_depth, self._output_depth),
            dtype=self.default.float_dtype,
            initializer=self.default.weight_initializer,
            regularizer=self.default.weight_regularizer
        )
        self._bias = Bias.instance(
            shape=(self._output_depth,),
            dtype=self.default.float_dtype,
            initializer=self.default.bias_initializer,
            regularizer=self.default.bias_regularizer
        )

    def _setup(self, x):
        return tf.tensordot(x, self._weight, axes=1) + self._bias

    @property
    def weight(self):
        return self._weight

    @property
    def bias(self):
        return self._bias


class LinearBNS(
    Linear, name='linear_bn_scale'
):
    def __init__(self, input_depth, output_depth, **kwargs):
        super(LinearBNS, self).__init__(input_depth, output_depth, **kwargs)

    def _build(self):
        self._bns = BatchNormWithScale([0], (self._output_depth,))

    def _setup(self, x):
        x = super(LinearBNS, self)._setup(x)
        return self._bns(x)

    @property
    def bns(self):
        return self._bns


class DeepLinear(
    DeepWidget, name='deep_linear', block=Linear
):
    def __init__(self, *depths, block=None, **kwargs):
        super(DeepLinear, self).__init__(block, **kwargs)
        assert len(depths) >= 2, 'At least two depths should be given'
        self._depths = depths

    def _build(self):
        self._layers = [
            self._block(input_depth, output_depth) for input_depth, output_depth in zip(self._depths, self._depths[1:])
        ]

    @property
    def depths(self):
        return self._depths
