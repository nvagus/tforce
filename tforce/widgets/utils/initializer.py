#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/8/17-10:43 AM
# :package: tforce.widgets.utils

import numpy as np
import tensorflow as tf

from ...core import DefaultChain
from ...core import Widget


class Initializer(Widget, name='initializer'):
    def __init__(self):
        super(Initializer, self).__init__()
        if not hasattr(self, '_f'):
            self._f = self.default.call

    def setup(self, shape, dtype):
        return super(Initializer, self).setup()

    def _setup(self, shape, dtype):
        return self._f(shape=shape, dtype=dtype)


class ZerosInitializer(Initializer, name='zeros_initializer', call=tf.zeros):
    pass


class OnesInitializer(Initializer, name='ones_initializer', call=tf.ones):
    pass


class VarianceScaling(DefaultChain, fin=1., fout=1., scale=1.):
    @classmethod
    def get_variance(cls, shape):
        prod = np.prod(shape[:-2])
        size_in = prod * shape[-2]
        size_out = prod * shape[-1]
        ret = np.sqrt(cls.default.scale / (cls.default.fin * size_in + cls.default.fout * size_out))
        return ret

    def __new__(cls, shape, dtype):
        raise NotImplementedError()


class HeNormal(VarianceScaling, fin=1., fout=0., scale=2.):
    def __new__(cls, shape, dtype):
        return tf.truncated_normal(stddev=cls.get_variance(shape), shape=shape, dtype=dtype)


class HeNormalInitializer(Initializer, name='he_normal_initializer', call=HeNormal):
    pass


class HeUniform(VarianceScaling, fin=1., fout=0., scale=2.):
    def __new__(cls, shape, dtype):
        return tf.random_uniform(shape, -3 * cls.get_variance(shape), 3 * cls.get_variance(shape))


class HeUniformInitializer(Initializer, name='he_uniform_initializer', call=HeUniform):
    pass


class LecunNormal(VarianceScaling, fin=1., fout=0., scale=1.):
    def __new__(cls, shape, dtype):
        return tf.truncated_normal(stddev=cls.get_variance(shape), shape=shape, dtype=dtype)


class LecunNormalInitializer(Initializer, name='lecun_normal_initializer', call=LecunNormal):
    pass


class LecunUniform(VarianceScaling, fin=1., fout=0., scale=1.):
    def __new__(cls, shape, dtype):
        return tf.random_uniform(shape, -3 * cls.get_variance(shape), 3 * cls.get_variance(shape))


class LecunUniformInitializer(Initializer, name='lecun_uniform_initializer', call=LecunUniform):
    pass


class GlorotNormal(VarianceScaling, fin=1., fout=1., scale=2.):
    def __new__(cls, shape, dtype):
        return tf.truncated_normal(stddev=cls.get_variance(shape), shape=shape, dtype=dtype)


class GlorotNormalInitializer(Initializer, name='glorot_normal_initializer', call=GlorotNormal):
    pass


class GlorotUniform(VarianceScaling, fin=1., fout=1., scale=2.):
    def __new__(cls, shape, dtype):
        return tf.random_uniform(shape, -3 * cls.get_variance(shape), 3 * cls.get_variance(shape))


class GlorotUniformInitializer(Initializer, name='glorot_uniform_initializer', call=GlorotUniform):
    pass
