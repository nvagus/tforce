#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/6/17-7:43 PM
# :package: tforce.test

import code

import tforce as t4

if __name__ == '__main__':
    default = t4.Default()
    default.a = 12
    default.b = '21'
    default.default = {1: 2, 2: '1'}
    copy = default()
    print(default.a, copy.a)
    default.a = 36
    print(default.a, copy.a)
    copy.f = t4
    print(copy.f)
    code.interact(local=locals())
