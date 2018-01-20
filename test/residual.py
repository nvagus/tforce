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

        t4.Regularizer.default.rate = 0.0005

        dropout = t4.Dropout(keep_prob=0.7)
        conv = t4.ConvBNS(3, 16, 3, 3, 1, 1)
        residual = t4.DeepResidualConv(
            (32, 6), (64, 6), (128, 6),
            input_channel=16, block=t4.SimpleResidualConv
        )
        lin = t4.Linear(128, 10)

        dense = t4.image.to_dense(image)
        dense = t4.image.randomize_shift(dense, 10, 5, 5)

        dense = conv(dense)
        dense = t4.relu(dense)

        dense = residual(dense, t4.relu, dropout=dropout)
        dense = t4.batch_normalization(dense)
        dense = t4.relu(dense)

        flat = t4.flat_pool(dense)
        pred = tf.nn.softmax(lin(flat))

        loss = t4.categorical_cross_entropy_loss(pred, t4.OneHot(10)(label))
        acc = t4.correct_prediction(t4.argmax(pred), label)

        loss_ma = t4.MovingAverage()(loss)
        acc_ma = t4.MovingAverage()(acc)

        self.lr = lr = tf.placeholder(t4.Widget.default.float_dtype)
        regularizers = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        optimizer = tf.train.MomentumOptimizer(lr, 0.9).minimize(loss + regularizers)
        
        self.train = self._add_slot(
            'train',
            outputs=(loss_ma, acc_ma, regularizers),
            updates=optimizer,
            givens={
                dropout.default.keep: True
            }
        )

        self.valid = self._add_slot(
            'valid',
            outputs=(loss, acc),
            givens={
                t4.BatchNorm.default.shift: False,
                dropout.default.keep: True
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
        'image', 'label',  dequeue_batch=128
    )
    model.setup(stream)
    lr = 0.1
    with model.using_workers():
        for _ in range(200):
            if _ in [60, 120, 160]:
                lr *= 0.2
            stream.option = 'train'
            t4.trainer.Alice(model.train).run(
                stream.data['train'].size // stream.batch_size, 1, givens={model.lr: lr}
            )
            stream.option = 'valid'
            t4.trainer.Bob(model.valid).run(
                stream.data['valid'].size // stream.batch_size
            )
    return 0


if __name__ == '__main__':
    exit(main())
