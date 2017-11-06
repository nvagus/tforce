#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/6/17-8:42 PM
# :package: tforce.test

import code
import tforce as t4


class A(t4.Scope, name='A'):
    def __init__(self):
        pass


if __name__ == '__main__':
    a = A()
    b = A(name='b')
    code.interact(local=locals())
