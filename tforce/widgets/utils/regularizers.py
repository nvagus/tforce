#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/8/17-10:20 AM
# :package: tforce.widgets.utils

import tensorflow as tf
from ...core import DefaultChain

class Regularizer(DefaultChain):
    def __init__(self, rate):
        self._rate = rate
        if not hasattr(self, '_f'):
            self._f = self.default.call

    def _setup(self, *args, **kwargs):
        loss = self._rate * self._f(*args, **kwargs)
