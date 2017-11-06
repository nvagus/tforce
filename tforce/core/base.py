#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/6/17-2:54 PM
# :package: tforce.core

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
            else:
                self._obj.obj = val

        def copy(self):
            return self.__class__(self._obj, const=True)

        __copy__ = copy

    def __getattr__(self, item):
        return self[item].obj

    def __setattr__(self, key, value):
        if isinstance(value, Default.Entry):
            self[key] = value
        elif key in self:
            self[key].obj = value
        else:
            self[key] = Default.Entry(value)

    def copy(self):
        default = self.__class__()
        for key, value in self.items():
            default[key] = value.copy()
        return default

    __copy__ = copy


class Scope(object):
    __scopes__ = {}
    default = Default()

    def __init_subclass__(cls, **param):
        cls.default = cls.default.copy()
        for key, val in param.items():
            setattr(cls.default, key, val)

        def __init__(self, *args, **kwargs):
            graph = tf.get_default_graph()
            if graph not in Scope.__scopes__:
                Scope.__scopes__[graph] = set()
            scopes = Scope.__scopes__[graph]
            outer_scope = graph.get_name_scope()
            name = kwargs.pop('name') if 'name' in kwargs else cls.default.name
            count = 0
            while True:
                self._name = '{}_{}'.format(name, count) if count else name
                self._scope = '{}/{}/'.format(outer_scope, self._name) if outer_scope is not '' \
                    else f'{self._name}/'
                if self._scope not in scopes:
                    scopes.add(self._scope)
                    break
                count += 1
            self._built = False
            self.__init__(*args, **kwargs)
            self.build()

        cls.__init__ = __init__

    @property
    def name(self):
        return self._name

    @property
    def _scope(self):
        return self._scope
