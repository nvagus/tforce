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

        image = tf.image.resize_images(image, (224, 224))
        image = t4.image.to_dense(image)
        from .tfmodel import imagenet_resnet_v2
        # training = tf.placeholder_with_default(True, ())
        # network = imagenet_resnet_v2(50, 1001)
        # cate = network(image, training)
        # import pprint
        # pprint.pprint(self.global_variables, indent=2)

        def resnet_model_fn(features, labels, mode, *unused):
            network = imagenet_resnet_v2(50, 1001)
            logits = network(
                inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

            predictions = {
                'classes': tf.argmax(logits, axis=1),
                'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
            }

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss
            )

        tf.estimator.Estimator(model_fn=resnet_model_fn, model_dir='../resnet')
        print(self.global_variables)
        import code
        code.interact(local=locals())
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
