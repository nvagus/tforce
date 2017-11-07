#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/6/17-8:42 PM
# :package: tforce.test

import code
import tforce as t4


class A(t4.Scope, name='A', parent='Scope', k=1):
    def __init__(self, **kwargs):
        super(A, self).__init__()
        print(f'init {self._name} in A')

    def _build(self):
        print(f'building {self._name}')


class B(A, name='B', parent='A'):
    def __init__(self, **kwargs):
        print(f'init {self._name} in B')
        super(B, self).__init__()


if __name__ == '__main__':
    a = A()
    b = A(name='b')
    c = B()
    code.interact(local=locals())
