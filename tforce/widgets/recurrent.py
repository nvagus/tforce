#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 3/3/18-9:32 PM
# :package: tforce.widgets.recurrent

import tensorflow as tf

from .utils import Layer
from .utils import OrthogonalInitializer
from ..core import Widget


class Cell(Layer, weight_initializer=OrthogonalInitializer):
    def __init__(self, input_size, output_size, state_size):
        super(Cell, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._state_size = state_size

    @property
    def input_size(self):
        return self._input_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def init_params(self, n):
        raise NotImplementedError


class GRUCell(Cell):
    def __init__(self, input_size, output_size, state_size=None):
        super(GRUCell, self).__init__(input_size, output_size, output_size or state_size)

    def _build(self):
        self._weight_xr = self._weight_variable((self._input_size, self._state_size))
        self._weight_xz = self._weight_variable((self._input_size, self._state_size))
        self._weight_xs = self._weight_variable((self._input_size, self._state_size))
        self._weight_sr = self._weight_variable((self._state_size, self._state_size))
        self._weight_sz = self._weight_variable((self._state_size, self._state_size))
        self._weight_ss = self._weight_variable((self._state_size, self._state_size))

    def _setup(self, state, x, *calls):
        reset = tf.sigmoid(tf.tensordot(x, self._weight_xr, axes=1) + tf.tensordot(state, self._weight_sr, axes=1))
        update = tf.sigmoid(tf.tensordot(x, self._weight_xz, axes=1) + tf.tensordot(state, self._weight_sz, axes=1))
        hidden = tf.tensordot(x, self._weight_xs, axes=1) + tf.tensordot(state, self._weight_ss, axes=1) * reset
        for call in calls:
            hidden = call(hidden)
        hidden = (1 - update) * hidden + update * state
        hidden.set_shape(state.get_shape())
        return hidden, hidden

    def init_params(self, n):
        x = tf.fill([n], tf.constant(0., self.default_float_dtype))
        return tf.map_fn(lambda x: tf.zeros(self._state_size, self.default_float_dtype), x), \
               tf.map_fn(lambda x: tf.zeros(self._output_size, self.default_float_dtype), x)


class Recur(Widget, block=GRUCell):
    def __init__(self, input_size, output_size, state_size=None, block=None):
        super(Recur, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._state_size = state_size
        self._block = block or self.default.block

    def _build(self):
        self._cell = self._block(self._input_size, self._output_size, self._state_size)

    def _setup(self, x, *calls):
        state, output = tf.scan(
            lambda s, i: self._cell(s[0], i, *calls),
            elems=tf.transpose(x, (1, 0, 2)),
            initializer=self._cell.init_params(tf.shape(x)[0])
        )
        return tf.transpose(output, (1, 0, 2))

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def block(self):
        return self._block


class Loop(Widget, block=GRUCell):
    def __init__(self, input_size, output_size, state_size=None, block=None):
        super(Loop, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._state_size = state_size
        self._block = block or self.default.block

    def _build(self):
        self._cell = self._block(self._input_size, self._output_size, self._state_size)

    def _setup(self, n, state, *calls):
        _, output_init = self._cell.init_params
        state, output = tf.scan(
            lambda s, i: self._cell(s[0], s[1], *calls),
            elems=tf.fill(n, tf.constant(0., self.default_float_dtype)),
            initializer=(state, output_init)
        )
        return tf.transpose(output, (1, 0, 2))

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def block(self):
        return self._block
