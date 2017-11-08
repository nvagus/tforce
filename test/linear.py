#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/8/17-5:19 PM
# :package: tforce.test

import code

import tensorflow as tf

import tforce as t4

if __name__ == '__main__':
    with tf.Session() as sess:
        x = tf.constant([1., 2., 3.], dtype=t4.Widget.default.float_dtype)
        lin = t4.Linear(3, 5)
        y = lin(x)
        sess.run(tf.global_variables_initializer())
        print(sess.run(y))
        print(
            lin.global_variables,
            lin.trainable_variables,
            lin.weight_variables,
            lin.bias_variables,
            lin.losses,
            lin.summaries,
            sep='\n'
        )
        code.interact(local=locals())
