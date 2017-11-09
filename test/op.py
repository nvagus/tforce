#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/9/17-10:36 AM
# :package: tforce.test

import code

import tensorflow as tf

import tforce as t4


@t4.Widget.from_op
def test(x, a=1, b=2):
    return x * a + b


@t4.Widget.from_op
def test2(x, y, c=3):
    return x - y * c


if __name__ == '__main__':
    with tf.Session() as sess:
        x = tf.constant(20)
        y = test(x)
        print(sess.run(y))
        z = test2(x, y)
        print(sess.run(z))

        code.interact(local=locals())
