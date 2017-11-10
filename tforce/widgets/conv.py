#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/10/17-9:56 AM
# :package: tforce.widgets

import tensorflow as tf

from ..core import Widget, DeepWidget

from .utils import HeNormalInitializer, ZerosInitializer

from .utils import L2Regularizer, NoRegularizer
from .utils import Filter, Bias
from .utils import BatchNormWithScale


class Conv(
    Widget, name='conv',
    filter_height=5, filter_width=5, stride_height=2, stride_height=2,
    filter_initializer=HeNormalInitializer, filter_regularizer=L2Regularizer,
    bias_initializer=ZerosInitializer, bias_regularizer=NoRegularizer
):
    def __init__(
            self, input_channel, output_channel,
            filter_height=None, filter_width=None, stride_height=None, stride_width=None,
            **kwargs
    ):
        super(Conv, self).__init__(**kwargs)
        self._input_channel = input_channel
        self._output_channel = output_channel
        self._filter_height = filter_height or self.default.filter_height
        self._filter_width = filter_width or self.default.filter_width
        self._stride_height = stride_height or self.default.stride_height
        self._stride_width = stride_width or self.default.stride_width

    def _build(self):
        self._filter = Filter(
            shape=(self._filter_height, self._filter_width, self._input_channel, self._output_channel),
            dtype=self.default.float_dtype,
            initializer=self.default.filter_initializer,
            regularizer=self.default.filter_regularizer
        )
        self._bias = Bias(
            shape=(self._output_channel,),
            dtype=self.default.float_dtype,
            initializer=self.default.bias_initializer,
            regularizer=self.default.bias_regularizer
        )

    def _setup(self, x):
        return tf.nn.conv2d(
            input=x,
            filter=self._filter,
            strides=(1, self._stride_height, self._stride_width, 1),
            padding='SAME',
            data_format='NHWC'
        ) + self._bias

    @property
    def filter(self):
        return self._filter

    @property
    def bias(self):
        return self._bias

    @property
    def filter_height(self):
        return self._filter_height

    @property
    def filter_width(self):
        return self._filter_width

    @property
    def stride_height(self):
        return self._stride_height

    @property
    def stride_width(self):
        return self._stride_width


class ConvBNS(Conv, name='stride_conv_bn_scale'):
    def __init__(
            self, input_channel, output_channel,
            filter_height=None, filter_width=None, stride_height=None, stride_width=None,
            **kwargs
    ):
        super(ConvBNS, self).__init__(
            input_channel, output_channel,
            filter_height, filter_width, stride_height, stride_width,
            **kwargs
        )

    def _build(self):
        super(ConvBNS, self)._build()
        self._bns = BatchNormWithScale()

    def _setup(self, x):
        x = super(ConvBNS, self)._setup(x)
        return self._bns(x)

    @property
    def bns(self):
        return self._bns


class DeepConv(DeepWidget, name='deep_conv', block=Conv):
    def __init__(
            self, *channels,
            filter_height=None, filter_width=None, stride_height=None, stride_width=None,
            block=None, **kwargs
    ):
        super(DeepConv, self).__init__(block, **kwargs)
        self._channels = channels
        self._filter_height = filter_height or self._block.default.stride_height
        self._filter_width = filter_width or self._block.default.stride_width
        self._stride_height = stride_height or self._block.default.stride_height
        self._stride_width = stride_width or self._block.default.stride_width

    def _build(self):
        self._layers = [
            self._block(
                input_channel, output_channel,
                self._filter_height, self._filter_width, self._stride_height, self._stride_width
            ) for input_channel, output_channel in zip(self._channels, self._channels[1:])
        ]

    @property
    def channels(self):
        return self._channels

    @property
    def filter_height(self):
        return self._filter_height

    @property
    def filter_width(self):
        return self._filter_width

    @property
    def stride_height(self):
        return self._stride_height

    @property
    def stride_width(self):
        return self._stride_width


class ResidualConv(
    DeepWidget, name='residual_conv', block=ConvBNS,
    filter_height=3, filter_width=3, stride_height=2, stride_height=2
):
    def __init__(
            self, input_channel, output_channel,
            filter_height=None, filter_width=None, stride_height=None, stride_width=None,
            block=None, **kwargs
    ):
        super(ResidualConv, self).__init__(block, **kwargs)
        self._input_channel = input_channel
        self._output_channel = output_channel
        self._filter_height = filter_height or self.default.filter_height
        self._filter_width = filter_width or self.default.filter_width
        self._stride_height = stride_height or self.default.stride_height
        self._stride_width = stride_width or self.default.stride_width

        def shortcut(x):
            return x

        self._shortcut = shortcut

    def _build(self):
        raise NotImplementedError()

    def _setup(self, x, *calls):
        y = super(ResidualConv, self)._setup(x, *calls)
        return self._shortcut(x) + y

    @property
    def shortcut(self):
        return self._shortcut

    @property
    def input_channel(self):
        return self._input_channel

    @property
    def output_channel(self):
        return self._output_channel

    @property
    def filter_height(self):
        return self._filter_height

    @property
    def filter_width(self):
        return self._filter_width

    @property
    def stride_height(self):
        return self._stride_height

    @property
    def stride_width(self):
        return self._stride_width


class SimpleResidualConv(ResidualConv, name='simple_residual_conv'):
    def __init__(
            self, input_channel, output_channel,
            filter_height=None, filter_width=None, stride_height=None, stride_width=None,
            block=None, **kwargs
    ):
        super(SimpleResidualConv, self).__init__(
            input_channel, output_channel,
            filter_height, filter_width, stride_height, stride_width,
            block, **kwargs
        )

    def _build(self):
        self._layers = [
            self._block(self._input_channel, self._input_channel, self._filter_height, self._filter_width, 1, 1),
            self._block(self._input_channel, self._input_channel, self._filter_height, self._filter_width,
                        self._stride_height, self._stride_width),
        ]
        if self._input_channel == self._output_channel or self._stride_height * self.stride_width == 1:
            self._shortcut = self._block(
                self._input_channel, self._output_channel, 1, 1, self._stride_height, self._stride_width
            )


class BottleNeckResidualConv(ResidualConv, name='bottle_neck_residual_conv', rate=4):
    def __init__(
            self, input_channel, output_channel,
            filter_height=None, filter_width=None, stride_height=None, stride_width=None,
            block=None, **kwargs
    ):
        super(BottleNeckResidualConv, self).__init__(
            input_channel, output_channel,
            filter_height, filter_width, stride_height, stride_width,
            block, **kwargs
        )
        self._sample_channel = self._input_channel // self.default.rate

    def _build(self):
        self._layers = [
            self._block(self.input_channel, self._sample_channel, 1, 1, 1, 1),
            self._block(self._sample_channel, self._sample_channel, self._filter_height, self._filter_width, 1, 1),
            self._block(self._sample_channel, self._output_channel, 1, 1, self._stride_height, self._stride_width)
        ]
        if self._input_channel == self._output_channel or self._stride_height * self.stride_width == 1:
            self._shortcut = self._block(
                self._input_channel, self._output_channel, 1, 1, self._stride_height, self._stride_width
            )


class DeepResidualConv(DeepWidget, block=SimpleResidualConv):
    def __init__(
            self, *channels,
            filter_height=None, filter_width=None, stride_height=None, stride_width=None,
            block=None, **kwargs
    ):
        super(DeepResidualConv, self).__init__(block, **kwargs)
        self._channels = channels
        self._filter_height = filter_height or self._block.default.stride_height
        self._filter_width = filter_width or self._block.default.stride_width
        self._stride_height = stride_height or self._block.default.stride_height
        self._stride_width = stride_width or self._block.default.stride_width

    def _build(self):
        self._layers = [
            self._block(
                input_channel, output_channel,
                self._filter_height, self._filter_width, self._stride_height, self._stride_width
            ) for input_channel, output_channel in zip(self._channels, self._channels[1:])
        ]

    @property
    def channels(self):
        return self._channels

    @property
    def filter_height(self):
        return self._filter_height

    @property
    def filter_width(self):
        return self._filter_width

    @property
    def stride_height(self):
        return self._stride_height

    @property
    def stride_width(self):
        return self._stride_width
