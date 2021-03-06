#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 9/2/17-8:02 PM
# :package: tforce

import os

import click

from .trainer import Trainer
from .trainer import callback as _callback

"""
Sample:
    @t4.main.begin
    @t4.main.gpu()
    @t4.main.end
    def main():
        # Instantiate your model and run here
        pass
"""

begin = click.command()
end = _callback


def _set_gpu(_, __, value):
    os.environ['CUDA_VISIBLE_DEVICES'] = value
    return value


def _set_batch_size(_, __, value):
    Trainer.default.batch_size = value
    return value


def gpu(default):
    if callable(default):
        return click.option('-d', '--gpu', prompt='GPU?', callback=_set_gpu)(default)
    else:
        return click.option('-d', '--gpu', default=str(default), callback=_set_gpu)


def batch_size(default):
    if callable(default):
        return click.option('-b', '--batch_size', prompt='Batch Size?', type=int, callback=_set_batch_size)(default)
    else:
        return click.option('-b', '--batch_size', default=default, type=int, callback=_set_batch_size)


option = click.option
argument = click.argument
