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

        t4.ResidualConv.default.block = t4.ConvNoBias
        conv = t4.ConvNoBias(3, 64, 7, 7, 2, 2)

        residual = t4.DeepResidualConv(
            (256, 3), (512, 4), (1024, 6), (2048, 3),
            input_channel=64, block=t4.BottleNeckResidualConvOS
        )
        lin = t4.Linear(2048, 90)

        # image = tf.image.resize_images(image, [224, 224])
        mean = tf.Variable(tf.constant([128, 128, 128], dtype=self.default_float_dtype), trainable=False)
        dense = t4.image.to_dense(image, mean=mean, std=1)

        dense = conv(dense)
        dense = t4.batch_normalization(dense)
        dense = t4.relu(dense)
        dense = t4.max_pool(dense, 3, 3, 2, 2)
        dense = residual(dense, t4.batch_normalization, t4.relu)

        flat = t4.flat_pool(dense)
        pred = tf.nn.softmax(lin(flat))

        loss = t4.categorical_cross_entropy_loss(pred, t4.OneHot(90)(label))
        acc = t4.correct_prediction(t4.argmax(pred), label)

        with tf.device('/gpu:1'):
            optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        self.train = self._add_slot(
            'train',
            outputs=(loss, acc),
            updates=optimizer
        )

        self.valid = self._add_slot(
            'valid',
            outputs=(loss, acc),
            givens={
                t4.BatchNorm.default.shift: False
            }
        )


# python3 -m test.imagenet -d 0
@t4.main.begin
@t4.main.batch_size(16)
@t4.main.gpu
@t4.main.end
def main():
    model = Model()
    stream = t4.MultiNpzDataStream(
        {'train': '/data/plan/coco/coco-train.npz', 'valid': '/data/plan/coco/coco-valid.npz'},
        'image', 'label'
    )
    model.setup(stream)
    with model.using_workers(50):
        model.restore('/data/noise/models/core.npz')

        stream.selected = 'train'
        t4.trainer.Alice(model.train).run(800, 1)
        stream.selected = 'valid'
        t4.trainer.Bob(model.valid, t4.trainer.Alice.default.batch_size).run(100)
    return 0


if __name__ == '__main__':
    exit(main())
