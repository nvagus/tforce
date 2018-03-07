#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 3/3/18-9:32 PM
# :package: tforce.widgets.recurrent

import tensorflow as tf

from .utils import OrthogonalInitializer, NoRegularizer, OnesInitializer
from .utils import Weight, Bias
from ..core import Widget


class RecurCell(Widget, weight_initializer=OrthogonalInitializer, weight_regularizer=NoRegularizer):
    def __init__(self, input_size, state_size):
        super(RecurCell, self).__init__()
        self._input_size = input_size
        self._state_size = state_size

    def initialize(self, x, state=None):
        raise NotImplementedError

    @property
    def input_size(self):
        return self._input_size

    @property
    def state_size(self):
        return self._state_size


class GRUCell(RecurCell):
    def __init__(self, input_size, state_size):
        super(GRUCell, self).__init__(input_size, state_size)

    def _build(self):
        self._weight_input_update = Weight.instance(
            (self._input_size, self._state_size),
            initializer=self.default.weight_initializer,
            regularizer=self.default.weight_regularizer
        )
        self._weight_input_reset = Weight.instance(
            (self._input_size, self._state_size),
            initializer=self.default.weight_initializer,
            regularizer=self.default.weight_regularizer
        )
        self._weight_input_state = Weight.instance(
            (self._input_size, self._state_size),
            initializer=self.default.weight_initializer,
            regularizer=self.default.weight_regularizer
        )
        self._weight_state_update = Weight.instance(
            (self._state_size, self._state_size),
            initializer=self.default.weight_initializer,
            regularizer=self.default.weight_regularizer
        )
        self._weight_state_reset = Weight.instance(
            (self._state_size, self._state_size),
            initializer=self.default.weight_initializer,
            regularizer=self.default.weight_regularizer
        )
        self._weight_state_state = Weight.instance(
            (self._state_size, self._state_size),
            initializer=self.default.weight_initializer,
            regularizer=self.default.weight_regularizer
        )

    def _setup(self, x, state):
        shape = state.get_shape()
        update = tf.sigmoid(
            tf.tensordot(x, self._weight_input_update, axes=1) +
            tf.tensordot(state, self._weight_input_update, axes=1)
        )
        reset = tf.sigmoid(
            tf.tensordot(x, self._weight_input_reset, axes=1) +
            tf.tensordot(state, self._weight_state_update, axes=1)
        )
        hidden = tf.tanh(
            tf.tensordot(x, self._weight_input_state, axes=1) +
            tf.tensordot(state, self._weight_state_state, axes=1) * reset
        )
        state = (1 - update) * hidden + update * state
        state.set_shape(shape)
        return state

    def initialize(self, x, state=None):
        return state if state is not None else tf.map_fn(
            lambda _: tf.zeros(self._state_size, self.default_float_dtype),
            x
        )

    @property
    def weight_input_state(self):
        return self._weight_input_state

    @property
    def weight_input_reset(self):
        return self._weight_input_reset

    @property
    def weight_input_update(self):
        return self._weight_input_update

    @property
    def weight_state_state(self):
        return self._weight_state_state

    @property
    def weight_state_reset(self):
        return self._weight_state_reset

    @property
    def weight_state_update(self):
        return self._weight_state_update


class ATTCell(RecurCell, bias_initializer=OnesInitializer, bias_regularizer=NoRegularizer):
    def __init__(self, input_size, state_size):
        super(ATTCell, self).__init__(input_size, state_size)
        self._input_size, self._att_size = input_size

    def _build(self):
        pass

    def _setup(self):
        pass

    def initialize(self, x, state=None):
        pass

    @property
    def att_size(self):
        return self._att_size


class Recur(Widget, block=GRUCell):
    def __init__(self, input_size, state_size, block=None):
        super(Recur, self).__init__()
        self._input_size = input_size
        self._state_size = state_size
        self._block = block or self.default.block

    def _build(self):
        self._cell = self._block(self._input_size, self._state_size)

    def _setup(self, x, state=None):
        init = self._cell.initialize(x, state)
        state = tf.scan(self._cell, tf.transpose(x, (1, 0, 2)), init)
        return tf.transpose(state, (1, 0, 2))

    @property
    def input_size(self):
        return self._input_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def cell(self):
        return self._cell

    @property
    def block(self):
        return self._block
