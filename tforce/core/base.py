#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/6/17-2:54 PM
# :package: tforce.core

import contextlib
import inspect
import os

import numpy as np
import tensorflow as tf


class Default(dict):
    class Entry(object):
        class ObjWrapper(object):
            def __init__(self, obj):
                self.obj = obj

        def __init__(self, obj, const=False):
            self._obj = obj if isinstance(obj, Default.Entry.ObjWrapper) else Default.Entry.ObjWrapper(obj)
            self._const = const

        @property
        def obj(self):
            return self._obj.obj

        @obj.setter
        def obj(self, val):
            if self._const:
                self._obj = Default.Entry.ObjWrapper(val)
                self._const = False
            else:
                self._obj.obj = val

        @property
        def const(self):
            return self._const

        def copy(self):
            return self.__class__(self._obj, const=True)

        __copy__ = copy

        def __str__(self):
            return str(self._obj.obj)

        def __repr__(self):
            return repr(self._obj.obj)

    def __init__(self):
        super(Default, self).__init__()
        self._active = False

    def __getattr__(self, item):
        return self[item].obj

    def __setattr__(self, key, value):
        if key in ['_active']:
            self.__dict__[key] = value
        elif isinstance(value, Default.Entry):
            self[key] = value
        elif key in self:
            if not self._active and self[key].const:
                raise PermissionError('Invalid Assignment')
            self[key].obj = value
        else:
            self[key] = Default.Entry(value)

    @property
    @contextlib.contextmanager
    def active(self):
        self._active = True
        try:
            yield
        finally:
            self._active = False

    def copy(self):
        default = self.__class__()
        for key, value in self.items():
            default[key] = value.copy()
        return default

    __copy__ = copy


class DefaultChain(object):
    default = Default()

    def __init_subclass__(cls, **kwargs):
        cls.default = cls.default.copy()
        with cls.default.active:
            for key, val in kwargs.items():
                setattr(cls.default, key, val)


class Scope(DefaultChain):
    __scopes__ = {}

    def __init__(self, **kwargs):
        if not issubclass(self.__class__, Scope):
            self._scope = ''
            self._name = ''

    def __init_subclass__(cls, **params):
        super(Scope, cls).__init_subclass__(**params)
        init = cls.__init__

        def __init__(self, *args, **kwargs):
            if hasattr(self, '_name'):
                init(self, *args, **kwargs)
            else:
                graph = tf.get_default_graph()
                if graph not in Scope.__scopes__:
                    Scope.__scopes__[graph] = set()
                scopes = Scope.__scopes__[graph]
                outer_scope = graph.get_name_scope()
                name = kwargs.pop('name') if 'name' in kwargs else cls.default.name
                count = 0
                while True:
                    self._name = f'{name}_{count}' if count else name
                    self._scope = f'{outer_scope}/{self._name}/' if outer_scope is not '' else f'{self._name}/'
                    if self._scope not in scopes:
                        scopes.add(self._scope)
                        break
                    count += 1
                init(self, *args, **kwargs)
                self.build()

        cls.__init__ = __init__

    def build(self):
        with tf.variable_scope(self._name), tf.name_scope(self._scope):
            self._build()

    def _build(self):
        pass

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


class Widget(Scope, name='widget', float_dtype=tf.float32, int_dtype=tf.int64):
    def __init__(self, **kwargs):
        super(Widget, self).__init__(**kwargs)

    def setup(self, *args, **kwargs):
        with tf.variable_scope(self._name), tf.name_scope(self._scope):
            return self._setup(*args, **kwargs)

    def _setup(self, *args, **kwargs):
        pass

    __call__ = setup

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

        class Op(Widget, name=f.__name__, **defaults):
            def __init__(self, **kwargs):
                super(Op, self).__init__(**kwargs)
                self.f = f

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


class DeepWidget(Widget, name='deep_widget', block=Widget):
    def __init__(self, block=None, **kwargs):
        super(DeepWidget, self).__init__(**kwargs)
        self._layers = []
        self._block = block or self.default.block

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
