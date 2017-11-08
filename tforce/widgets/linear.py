#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/8/17-4:49 PM
# :package: tforce.widgets

import tensorflow as tf

from .utils import HeNormalInitializer, ZerosInitializer
from .utils import L2Regularizer, NoRegularizer
from .utils import Weight, Bias
from ..core import Widget


class Linear(
    Widget, name='linear',
    weight_initializer=HeNormalInitializer, weight_regularizer=L2Regularizer,
    bias_initializer=ZerosInitializer, bias_regularizer=NoRegularizer
):
    def __init__(self, input_depth, output_depth):
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
