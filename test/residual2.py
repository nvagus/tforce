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
        label = data['label'] - 1

        k = 2
        n = 4
        t4.L2Regularizer.default.rate = 0.0005

        t4.SimpleResidualConv.default.block = t4.ConvNoBias
        conv = t4.Conv(3, 16, 3, 3, 1, 1)
        residual = t4.DeepResidualConv(
                (16 * k, n), (32 * k, n), (64 * k, n), (128 * k, n),
                input_channel=16, block=t4.SimpleResidualConv
            )

        lin = t4.Linear(128 * k, 1000)

        shift = tf.placeholder_with_default(True, ())
        image = t4.image.randomize_flip(image, 0.5, switch=shift)
        image = t4.image.randomize_crop(image, switch=shift)
        dense = t4.image.to_dense(image, std=64)

        dense = conv(dense)
        dense = t4.batch_normalization(dense)
        dense = t4.relu(dense)

        dense = residual(dense, t4.batch_normalization, t4.relu)

        flat = t4.flat_pool(dense)
        pred = tf.nn.softmax(t4.batch_normalization(lin(flat)))

        loss = t4.categorical_cross_entropy_loss(pred, t4.OneHot(1000)(label),
                                                 with_false=False
                                                 )

        acc = t4.correct_prediction(t4.argmax(pred), label)

        loss_ma = t4.MovingAverage()(loss)
        acc_ma = t4.MovingAverage()(acc)

        self.lr = lr = tf.placeholder(t4.Widget.default.float_dtype)
        regularizers = tf.reduce_sum(self.losses)

        optimizer = tf.train.MomentumOptimizer(lr, 0.9).minimize(loss
                                                                  + regularizers,
                                                                 colocate_gradients_with_ops=True)

        self.train = self._add_slot(
            'train',
            outputs=(loss_ma, acc_ma, regularizers),
            updates=optimizer
        )

        self.valid = self._add_slot(
            'valid',
            outputs=(loss, acc),
            givens={
                t4.BatchNorm.default.shift: False,
                shift: False
            }
        )


# python3 -m test.residual -d 0
@t4.main.begin
@t4.main.batch_size(128)
@t4.main.gpu
@t4.main.end
def main():
    model = Model()
    stream = t4.MultiNpzDataStream(
        {'train': '/data/plan/imagenet-64/imagenet-64-train.npz',
         'valid': '/data/plan/imagenet-64/imagenet-64-valid.npz'},
        'image', 'label'
    )
    model.setup(stream)
    lr = 0.01
    with model.using_workers(20):
        for epoch in range(50):
            if epoch in [10, 20, 30, 40]:
                lr *= 0.2
            print(f'epoch {epoch} ' + '-' * 100)
            stream.selected = 'train'
            t4.trainer.Alice(model.train).run(
                stream.data['train'].size // t4.trainer.Alice.default.batch_size, 1, givens={model.lr: lr}
            )
            stream.selected = 'valid'
            t4.trainer.Bob(model.valid, 128).run(80, highlight=True)
    return 0


if __name__ == '__main__':
    exit(main())
