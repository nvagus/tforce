#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/8/17-10:43 AM
# :package: tforce.widgets.utils

import numpy as np
import tensorflow as tf

from ...core import DefaultChain
from ...core import Widget


class Initializers(Widget, name='initializer'):
    def __init__(self):
        super(Initializers, self).__init__()
        if not hasattr(self, '_f'):
            self._f = self.default.call

    def setup(self, shape, dtype):
        return super(Initializers, self).setup()

    def _setup(self, shape, dtype):
        return self._f(shape=shape, dtype=dtype)


class ZerosInitializer(Initializers, name='zeros_initializer', call=tf.zeros):
    pass


class OnesInitializer(Initializers, name='ones_initializer', call=tf.ones):
    pass


class VarianceScaling(DefaultChain, fin=1., fout=1., scale=1.):
    @classmethod
    def get_variance(cls, shape):
        prod = np.prod(shape[:-2])
        size_in = prod * shape[-2]
        size_out = prod * shape[-1]
        return np.sqrt(cls.default.scale / (cls.default.fin * size_in + cls.default.fout * size_out))

    def __new__(cls, shape, dtype):
        raise NotImplementedError()


class HeNormal(VarianceScaling, fin=1., fout=0., scale=2.):
    def __new__(cls, shape, dtype):
        return tf.truncated_normal(stddev=cls.get_variance(shape), shape=shape, dtype=dtype)


class HeNormalInitializer(Initializers, name='he_normal_initializer', call=HeNormal):
    pass
