#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 2017/8/25-16:53
# :package: tforce

import inspect
import os

import numpy as np

from ..core import Slot, Root
from ..core.private import make_iterable as _make_iterable


class _Console(object):
    def __init__(self, enter_code, exit_code):
        self._enter_code = enter_code
        self._exit_code = exit_code

    def __enter__(self):
        print(self._enter_code, end='', flush=True)

    def __exit__(self, *unused):
        print(self._exit_code, end='', flush=True)


class _HideCursor(_Console):
    def __init__(self):
        super(_HideCursor, self).__init__('\33[?25l', '\33[?25h')


class _Style(_Console):
    fore = {
        'black': ';30',
        'red': ';31',
        'green': ';32',
        'yellow': ';33',
        'blue': ';34',
        'purple': ';35',
        'cyan': ';36',
        'white': ';37',
        None: ';'
    }

    back = {
        'black': ';40',
        'red': ';41',
        'green': ';42',
        'yellow': ';43',
        'blue': ';44',
        'purple': ';45',
        'cyan': ';46',
        'white': ';47',
        None: ''
    }

    mode = {
        'normal': 0,
        'bold': 1,
        'underline': 4,
        'blink': 5,
        'invert': 7,
        'hide': 8,
        None: 0
    }

    def __init__(self, mode=None, fore=None, back=None):
        super(_Style, self).__init__(f'\033[{_Style.mode[mode]}{_Style.fore[fore]}{_Style.back[back]}m', '\033[0m')


def _print_log(name, step, values, labels, fmt='{}={:<12.5f}', next_line=True):
    end = '\n' if next_line else '\r'
    pairs = zip(_make_iterable(labels), _make_iterable(values))
    print(f'{name} loop:{step:<10d}\t' + '\t'.join([fmt.format(*pair) for pair in pairs]), end=end,
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


class Trainer(Root, batch_size=100):
    """ Trainer is someone who helps for the training
    """

    def __init__(self, slot: Slot, batch_size=None):
        self._slot = slot
        self._result = None
        self._batch_size = batch_size or self.default.batch_size
        self._fields = {field: [] for field in slot.labels}

    def run(self, *args, **kwargs):
        raise NotImplementedError()

    def _add_log(self, result):
        for k, v in zip(_make_iterable(self._slot.labels), _make_iterable(result)):
            self._fields[k].append(v)

    @property
    def log(self):
        return {key: np.hstack(self._fields[key]) for key in self._fields}

    @property
    def result(self):
        return self._result

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
        callbacks = _make_iterable(callbacks)
        givens = givens or {}
        givens.update(self.slot.model.givens(self._batch_size))
        with _HideCursor():
            for i in range(1, steps + 1):
                self._result = result = self._slot(givens=givens)
                self._add_log(result)
                if i % log_step == 0:
                    _print_log(
                        self._slot.name,
                        self._slot.step,
                        result,
                        self._slot.labels,
                        next_line=i + log_step > steps
                    )
                for call in callbacks:
                    call(step=self._slot.step, result=result, givens=givens)
        return self


class Bob(Trainer):
    """ Bob does validation, who computing average performance of output targets.
    """

    def run(self, steps=1, highlight=False, givens=None):
        givens = givens or {}
        givens.update(self.slot.model.givens(self._batch_size))
        for _ in range(steps):
            self._result = result = self._slot(givens=givens)
            self._add_log(result)
        log = self.log
        self._result = result = [np.average(log[key]) for key in log]
        with _Style('bold', 'red') if highlight else _Style():
            _print_log(self._slot.name, self._slot.step, result, self._slot.labels, next_line=True)
        return self
