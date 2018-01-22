#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/14/17-3:49 PM
# :package: tforce.test

import tensorflow as tf

import tforce as t4


class Model(t4.Model):
    def __init__(self):
        super(Model, self).__init__()

    def _setup(self, data):
        image = data['image']
        label = data['label']

        t4.Conv.default.filter_initializer = t4.GlorotNormalInitializer
        conv = t4.DeepConv(
            1, 64, 1024
        )
        lin = t4.Linear(7 * 7 * 1024, 10)

        dense = t4.image.to_dense(image)
        flat = t4.image.to_flat(conv(dense, t4.relu))
        pred = tf.nn.softmax(lin(flat))

        loss = t4.categorical_cross_entropy_loss(pred, tf.one_hot(label, 10, dtype=t4.Widget.default.float_dtype))
        acc = t4.correct_prediction(tf.argmax(pred, axis=1), label)

        optimizer = tf.train.AdamOptimizer().minimize(loss)

        self.train = self._add_slot(
            'train',
            outputs=(loss, acc),
            updates=optimizer,
            summaries=lin.summaries
        )

        self.valid = self._add_slot(
            'valid',
            outputs=(loss, acc),
            # givens={t4.BatchNorm.default.shift: False}
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
        stream.selected = 'train'
        t4.trainer.Alice(model.train).run(600)
        stream.selected = 'valid'
        t4.trainer.Bob(model.valid, 10000).run()
    return 0


if __name__ == '__main__':
    exit(main())
