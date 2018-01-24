#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/6/17-2:54 PM
# :package: tforce.core


import inspect
import itertools
import os

import numpy as np
import tensorflow as tf


class Default(object):
    def __init__(self, __prior__=None, **kwargs):
        self.__prior__ = __prior__
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __call__(self, **kwargs):
        return Default(self, **kwargs)

    def __getattr__(self, name):
        if self.__prior__:
            return getattr(self.__prior__, name)
        else:
            raise AttributeError(f'{name} is not in this default chain')


class Root(object):
    default = Default('Root')

    def __init_subclass__(cls, **kwargs):
        cls.default = cls.default(**kwargs)


class Scope(type):
    __scopes__ = {}

    def __call__(cls, *args, **kwargs):
        if 'name' in kwargs:
            name = kwargs.pop('name')
        elif isinstance(cls, Root):
            if hasattr(cls.default, 'name'):
                name = cls.default.name
            else:
                name = cls.__name__
        else:
            name = cls.__name__

        obj = super(Scope, cls).__call__(*args, **kwargs)

        obj.__graph__ = graph = obj.__graph__ if hasattr(obj, '__graph__') else tf.get_default_graph()
        Scope.__scopes__[graph] = scopes = Scope.__scopes__.get(graph) or set()
        outer_scope = graph.get_name_scope()

        for count in itertools.count():
            _name = f'{name}_{count}' if count else name
            _scope = f'{outer_scope}/{_name}/' if outer_scope is not '' else f'{_name}/'
            if _scope not in scopes:
                scopes.add(_scope)
                return obj.build(_name, _scope)


class Widget(Root, metaclass=Scope, float_dtype=tf.float32, int_dtype=tf.int64):
    def __init__(self):
        super(Widget, self).__init__()
        self._name = None
        self._scope = None

    def build(self, name, scope):
        self._name = name
        self._scope = scope
        with tf.variable_scope(self._name), tf.name_scope(self._scope):
            self._build()
        return self

    def _build(self):
        pass

    def setup(self, *args, **kwargs):
        with tf.variable_scope(self._name), tf.name_scope(self._scope):
            return self._setup(*args, **kwargs)

    def _setup(self, *args, **kwargs):
        pass

    __call__ = setup

    @property
    def default_float_dtype(self):
        return self.default.float_dtype

    @property
    def default_int_dtype(self):
        return self.default.int_dtype

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def global_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self._scope)

    @property
    def trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self._scope)

    @property
    def weight_variables(self):
        return tf.get_collection(tf.GraphKeys.WEIGHTS, self._scope)

    @property
    def bias_variables(self):
        return tf.get_collection(tf.GraphKeys.BIASES, self._scope)

    @property
    def moving_average_variables(self):
        return tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, self._scope)

    @property
    def losses(self):
        return tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self._scope)

    @property
    def summaries(self):
        return tf.get_collection(tf.GraphKeys.SUMMARIES, self._scope)

    def save(self, filename):
        le = len(self._scope)
        variables = {x.name[le:]: x.eval() for x in self.global_variables}
        np.savez(filename, **variables)

    def restore(self, filename):
        le = len(self._scope)
        if os.path.isfile(filename):
            initials = np.load(filename)
            return [tf.assign(x, tf.constant(initials[x.name[le:]])) for x in self.global_variables]
        else:
            return tf.no_op()

    class Lazy(object):
        def __init__(self, cls, *args, **kwargs):
            self._cls = cls
            self._args = args
            self._kwargs = kwargs

        def __call__(self):
            return self._cls(*self._args, **self._kwargs)

    @classmethod
    def lazy(cls, *args, **kwargs):
        return cls.Lazy(cls, *args, **kwargs)

    @staticmethod
    def from_op(f):
        params = inspect.signature(f).parameters
        defaults = {key: val.default for key, val in params.items() if val.default is not inspect.Parameter.empty}

        class Op(Widget, **defaults):
            def __init__(self):
                super(Op, self).__init__()
                self.f = f
                self.__name__ = f.__name__

            def _setup(self, *args, **kwargs):
                for key, val in zip(params, args):
                    kwargs[key] = val
                for key, val in defaults.items():
                    if key not in kwargs:
                        kwargs[key] = val
                return self.f(**kwargs)

        def op_wrapper(*args, **kwargs):
            return Op()(*args, **kwargs)

        op_wrapper.__name__ = f.__name__
        op_wrapper.default = Op.default
        op_wrapper.Op = Op

        return op_wrapper

    @property
    def to_keras(self):
        _self = self

        import keras

        class KerasLayer(keras.layers.Layer):
            def __init__(self):
                super(KerasLayer, self).__init__()

            def call(self, *args, **kwargs):
                return _self(*args, **kwargs)

        return KerasLayer()


class DeepWidget(Widget, block=Widget):
    def __init__(self, block=None):
        super(DeepWidget, self).__init__()
        self._layers = []
        self._block = block or self.default.block

    def _build(self):
        self._layers = []

    def _setup(self, x, *calls):
        for layer in self._layers[:-1]:
            x = layer(x)
            for call in calls:
                x = call(x)
        return self._layers[-1](x)

    @property
    def layers(self):
        return self._layers

    @property
    def block(self):
        return self._block
