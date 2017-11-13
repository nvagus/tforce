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

        dense = t4.image.to_dense(image)
        flat = t4.image.to_flat(dense)
        pred = tf.nn.softmax(t4.Linear(28 * 28, 10)(flat))

        loss = t4.categorical_cross_entropy_loss(pred, tf.one_hot(label, 10, dtype=t4.Widget.default.float_dtype))
        acc = t4.correct_prediction(tf.argmax(pred, axis=1), label)

        optimizer = tf.train.AdamOptimizer().minimize(loss)

        self.train = self._add_slot(
            'train',
            outputs=(loss, acc),
            updates=optimizer
        )

        self.valid = self._add_slot(
            'valid',
            outputs=(loss, acc)
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
    model.setup(stream)
    with model.using_workers():
        stream.option = 'train'
        t4.trainer.Alice(model.train).run(1200, 1)
        stream.option = 'valid'
        t4.trainer.Bob(model.valid).run(200)
    return 0


if __name__ == '__main__':
    exit(main())
