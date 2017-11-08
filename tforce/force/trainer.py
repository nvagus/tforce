#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 2017/8/25-16:53
# :package: tforce

import inspect
import os

import numpy as np

from ..core import Slot
from ..core.private import make_iterable as _make_iterable


class _HideCursor(object):
    def __enter__(self):
        print('\33[?25l', end='', flush=True)

    def __exit__(self, *unused):
        print('\33[?25h', end='', flush=True)


def _print_log(name, step, values, labels, fmt='{}={:<15.5f}', next_line=True):
    end = '\n' if next_line else '\r'
    pairs = zip(_make_iterable(labels), _make_iterable(values))
    print('{} loop:{:<10d}\t'.format(name, step) + '\t'.join([fmt.format(*pair) for pair in pairs]), end=end,
          flush=True)


def callback(f):
    """ Ignore keyword arguments given to f that are not shown in the f's signature.
    """
    parameters = inspect.signature(f).parameters

    def _wrapper(**kwargs):
        invalid = [key for key in kwargs if key not in parameters]
        for key in invalid:
            kwargs.pop(key)
        return f(**kwargs)

    return _wrapper


class Trainer(object):
    """ Trainer is someone who helps for the training
    """

    def __init__(self, slot: Slot):
        self._slot = slot
        self._fields = {field: [] for field in slot.labels}

    def run(self, *args, **kwargs):
        raise NotImplementedError()

    def _add_log(self, result):
        for k, v in zip(_make_iterable(self._slot.labels), _make_iterable(result)):
            self._fields[k].append(v)

    @property
    def log(self):
        return {key: np.hstack(self._fields[key]) for key in self._fields}

    def save(self, filename):
        np.savez(filename, **self.log)

    def savefig(self, folder=None):
        import matplotlib.pyplot as plt
        for key, val in self.log.items():
            filepath = '{}.png'.format(os.path.join(folder, key) if folder else key)
            plt.plot(np.arange(len(val)), val)
            plt.savefig(filepath)
            plt.clf()

    __call__ = run

    @property
    def slot(self):
        return self._slot


class Alice(Trainer):
    """ Alice does common training process, prints logs, and allows you to use callbacks to apply custom commands.
    """

    def run(self, steps, log_step=10, givens=None, callbacks=None):
        assert steps % log_step == 0, 'Steps should be a multiple of the param log_step'
        callbacks = _make_iterable(callbacks)
        with _HideCursor():
            for i in range(1, steps + 1):
                result = self._slot(givens=givens)
                self._add_log(result)
                if i % log_step == 0:
                    _print_log(self._slot.name, self._slot.local_step, result, self._slot.labels, next_line=i == steps)
                for call in callbacks:
                    call(step=self._slot.local_step, result=result, givens=givens)
        return self


class Bob(Trainer):
    """ Bob does validation, who computing average performance of output targets.
    """

    def run(self, steps, givens=None):
        for _ in range(steps):
            result = self._slot(givens=givens)
            self._add_log(result)
        log = self.log
        result = tuple(
            np.average(log[key]) for key in self._slot.labels
        )
        _print_log(self._slot.name, self._slot.local_step, result, self._slot.labels, next_line=True)
        return self
