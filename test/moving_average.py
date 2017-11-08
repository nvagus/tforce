#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/8/17-10:04 PM
# :package: tforce.test

import code

import tensorflow as tf

import tforce as t4

if __name__ == '__main__':
    with tf.Session() as sess:
        ma = t4.MovingAverage(0.999, shape=())
        x = tf.placeholder(shape=(), dtype=tf.float32)
        y = ma(x)


        def run(n):
            return sess.run(y, feed_dict={x: n})


        sess.run(tf.global_variables_initializer())
        print(run(1.))
        code.interact(local=locals())
