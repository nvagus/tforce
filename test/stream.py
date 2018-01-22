#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/8/17-8:08 PM
# :package: tforce.test

import tensorflow as tf

import tforce as t4


class Model(t4.Model):
    def __init__(self):
        super(Model, self).__init__()

    def _setup(self, data):
        image = data['image']
        label = data['label']
        maxlab = tf.argmax(label)
        self.show = self._add_slot(
            'show',
            outputs=maxlab
        )


# python3 -m test.mnist -d 0
@t4.main.begin
@t4.main.gpu
@t4.main.end
def main():
    model = Model()
    stream = t4.MultiNpzDataStream(
        {'train': '/data/plan/mnist/mnist-train.npz', 'valid': '/data/plan/mnist/mnist-valid.npz'},
        'image', 'label'
    )
    # stream = t4.NpzDataStream('/data/plan/mnist/mnist-valid.npz', 'image', 'label')
    model.setup(stream)
    with stream.using_workers():
        stream.selected = 'train'
        print(model.sess.run(stream.batch['label'], feed_dict=stream.givens(1)))
        stream.selected = 'valid'
        print(model.sess.run(stream.batch['label'], feed_dict=stream.givens(10)))
    return 0


if __name__ == '__main__':
    exit(main())
