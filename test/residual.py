#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/14/17-3:49 PM
# :package: tforce.test


import tensorflow as tf

import tforce as t4


class _Model(t4.Model):
    def __init__(self):
        super(_Model, self).__init__()

    def _setup(self, data):
        image = data['image']
        label = data['label']

        conv = t4.ConvBNS(3, 64, 7, 7)
        residual = t4.DeepResidualConv(
            (128, 2), (256, 2), (512, 2), (1024, 2),
            input_channel=64, block=t4.SimpleResidualConv
        )
        lin = t4.Linear(1024, 10)

        dense = t4.image.to_dense(image)
        t4.BatchNorm.default.shift = t4.MovingAverage.new_shift()
        # pred = ResnetBuilder.build_resnet_18((3, 32, 32), 10)(dense)
        dense = conv(dense)
        dense = t4.relu(dense)
        dense = t4.max_pool(dense, 3, 3, 2, 2)
        dense = residual(dense, t4.relu)
        dense = t4.BatchNormWithScale([0, 1, 2], (1024,))(dense)
        dense = t4.relu(dense)
        flat = t4.flat_pool(dense)
        pred = tf.nn.softmax(lin(flat))

        loss = t4.categorical_cross_entropy_loss(pred, tf.one_hot(label, 10, dtype=t4.Widget.default.float_dtype))
        acc = t4.correct_prediction(tf.argmax(pred, axis=1), label)

        optimizer = tf.train.AdamOptimizer().minimize(loss)
        
        self.train = self._add_slot(
            'train',
            outputs=(loss, acc),
            updates=optimizer,
            # summaries=residual.summaries
        )

        self.valid = self._add_slot(
            'valid',
            outputs=(loss, acc),
            givens={
                t4.BatchNorm.default.shift: False,
            }
        )


# python3 -m test.mnist -d 0
@t4.main.begin
@t4.main.gpu
@t4.main.end
def main():
    model = _Model()
    stream = t4.MultiNpzDataStream(
        {'train': '/data/plan/cifar-10/cifar-10-train.npz', 'valid': '/data/plan/cifar-10/cifar-10-valid.npz'},
        'image', 'label'
    )
    model.setup(stream)
    with model.using_workers(), model.using_summaries():
        stream.option = 'train'
        t4.trainer.Alice(model.train).run(1000, 1)
        stream.option = 'valid'
        t4.trainer.Bob(model.valid).run(200)
    return 0


if __name__ == '__main__':
    exit(main())
