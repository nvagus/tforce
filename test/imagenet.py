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

        tf.train.import_meta_graph("../resnet/resnet_v1_50.ckpt")

        print(self._graph.)
        exit()


# python3 -m test.residual -d 0
@t4.main.begin
@t4.main.batch_size(128)
@t4.main.gpu
@t4.main.end
def main():
    model = Model()
    stream = t4.MultiNpzDataStream(
        {'train': '/data/plan/ilsvrc/ilsvrc-train.npz',
         'valid': '/data/plan/ilsvrc/ilsvrc-valid.npz'},
        'image', 'label'
    )
    model.setup(stream)

    # with model.using_workers():
    #     pass
    return 0


if __name__ == '__main__':
    exit(main())
