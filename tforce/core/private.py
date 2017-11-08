#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 2017/8/18-11:30
# :package: tforce.core

import collections
import functools

import tensorflow as tf


def astuple(f):
    def _wrapper(*args, **kwargs):
        return tuple(f(*args, **kwargs))

    return _wrapper


def iterable(obj):
    return isinstance(obj, collections.Iterable) and not isinstance(obj, tf.Tensor)


def make_iterable(obj):
    # obj is obj1, obj2, ... -> (obj1, obj2, ...)
    # obj is obj1            -> (obj1, )
    # obj is None            -> ()
    if iterable(obj):
        return tuple(obj)
    elif obj is None:
        return ()
    else:
        return obj,


def make_multiple(obj):
    # obj is (obj1, obj2, ...) -> (obj1, obj2, ...)
    # obj is (obj1, )          -> obj1
    # obj is ()                -> None
    assert iterable(obj), 'Only iterable objects can be made multiple'
    obj = tuple(obj)
    le = len(obj)
    if le > 1:
        return obj
    elif le == 1:
        return obj[0]
    else:
        return None


def inspect_names(frame, var_list):
    """
    Get the names of vars from a python run-time frame
    :param frame: the frame where the variables can be found
    :param var_list: the list of what to find
    :return: a list of names for each item in the var_list
    :raise: SyntaxError if there is any name not found
    """
    error_code = 'c1268a3929ee7fd5257d84c58a31b438'

    @astuple
    def _get_keys(d, l):
        k = list(None for _ in range(len(l)))
        for key, v in d.items():
            for i in range(len(l)):
                if l[i] is v:
                    k[i] = key
        if any(_k is None for _k in k):
            raise ValueError(error_code)
        return k

    assert isinstance(var_list, (list, tuple)), 'Param var_list should be iterable'
    if var_list:
        try:
            return tuple(_get_keys(frame.f_locals, var_list))
        except ValueError as e:
            if str(e).startswith(error_code):
                raise SyntaxError('You should pass non-temporary args')
    else:
        return ()
