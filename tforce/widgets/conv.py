#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 11/10/17-9:56 AM
# :package: tforce.widgets

import tensorflow as tf

from .utils import Filter, Bias
from .utils import HeNormalInitializer, ZerosInitializer
from .utils import L2Regularizer, NoRegularizer
from ..core import Widget, DeepWidget


class Conv(
    Widget,
    filter_height=5, filter_width=5, stride_height=2, stride_width=2, padding='SAME',
    filter_initializer=HeNormalInitializer, filter_regularizer=L2Regularizer,
    bias_initializer=ZerosInitializer, bias_regularizer=NoRegularizer
):
    def __init__(
            self, input_channel, output_channel,
            filter_height=None, filter_width=None, stride_height=None, stride_width=None,
    ):
        super(Conv, self).__init__()
        self._input_channel = input_channel
        self._output_channel = output_channel
        self._filter_height = filter_height or self.default.filter_height
        self._filter_width = filter_width or self.default.filter_width
        self._stride_height = stride_height or self.default.stride_height
        self._stride_width = stride_width or self.default.stride_width

    def _build(self):
        self._filter = Filter.instance(
            shape=(self._filter_height, self._filter_width, self._input_channel, self._output_channel),
            dtype=self.default.float_dtype,
            initializer=self.default.filter_initializer,
            regularizer=self.default.filter_regularizer
        )
        self._bias = Bias.instance(
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
            padding=self.default.padding,
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


class DeepConv(DeepWidget, block=Conv):
    def __init__(
            self, *channels,
            filter_height=None, filter_width=None, stride_height=None, stride_width=None,
            block=None
    ):
        super(DeepConv, self).__init__(block)
        self._channels = channels
        self._filter_height = filter_height or self._block.default.filter_height
        self._filter_width = filter_width or self._block.default.filter_width
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
    DeepWidget, block=Conv,
    filter_height=3, filter_width=3, stride_height=2, stride_width=2
):
    def __init__(
            self, input_channel, output_channel,
            filter_height=None, filter_width=None, stride_height=None, stride_width=None,
            block=None, norm=None
    ):
        super(ResidualConv, self).__init__(block)
        self._input_channel = input_channel
        self._output_channel = output_channel
        self._filter_height = filter_height or self.default.filter_height
        self._filter_width = filter_width or self.default.filter_width
        self._stride_height = stride_height or self.default.stride_height
        self._stride_width = stride_width or self.default.stride_width

        def shortcut(x):
            return x

        self._shortcut = shortcut

    def _setup(self, x, *calls, dropout=None, first=False):
        dropout = dropout or (lambda z: z)
        y = x
        for layer in self._layers:
            if not first or layer is not self._layers[0]:
                for call in calls:
                    y = call(y)
            y = layer(y)
            if layer is not self._layers[-1]:
                y = dropout(y)
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


class SimpleResidualConv(ResidualConv):
    def __init__(
            self, input_channel, output_channel,
            filter_height=None, filter_width=None, stride_height=None, stride_width=None,
            block=None
    ):
        super(SimpleResidualConv, self).__init__(
            input_channel, output_channel,
            filter_height, filter_width, stride_height, stride_width,
            block
        )

    def _build(self):
        self._layers = [
            self._block(self._input_channel, self._input_channel, self._filter_height, self._filter_width, 1, 1),
            self._block(self._input_channel, self._output_channel, self._filter_height, self._filter_width,
                        self._stride_height, self._stride_width),
        ]
        if self._input_channel != self._output_channel or self._stride_height * self.stride_width != 1:
            self._shortcut = self._block(
                self._input_channel, self._output_channel, 1, 1, self._stride_height, self._stride_width
            )


class BottleNeckResidualConv(ResidualConv, rate=4):
    def __init__(
            self, input_channel, output_channel,
            filter_height=None, filter_width=None, stride_height=None, stride_width=None,
            block=None
    ):
        super(BottleNeckResidualConv, self).__init__(
            input_channel, output_channel,
            filter_height, filter_width, stride_height, stride_width,
            block
        )
        self._sample_channel = self._input_channel // self.default.rate

    def _build(self):
        self._layers = [
            self._block(self.input_channel, self._sample_channel, 1, 1, 1, 1),
            self._block(self._sample_channel, self._sample_channel, self._filter_height, self._filter_width, 1, 1),
            self._block(self._sample_channel, self._output_channel, 1, 1, self._stride_height, self._stride_width)
        ]
        if self._input_channel != self._output_channel or self._stride_height * self.stride_width != 1:
            self._shortcut = self._block(
                self._input_channel, self._output_channel, 1, 1, self._stride_height, self._stride_width
            )


class DeepResidualConv(DeepWidget, block=SimpleResidualConv):
    def __init__(
            self, *channels,
            filter_height=None, filter_width=None, stride_height=None, stride_width=None,
            block=None, input_channel=None
    ):
        super(DeepResidualConv, self).__init__(block)
        self._channels = sum([[channel] * times for channel, times in channels], [])
        self._filter_height = filter_height or self._block.default.filter_height
        self._filter_width = filter_width or self._block.default.filter_width
        self._stride_height = stride_height or self._block.default.stride_height
        self._stride_width = stride_width or self._block.default.stride_width
        self._input_channel = input_channel or self._channels[0]

    def _build(self):
        input_channels = [self._input_channel] + self._channels
        first_layer = True
        for input_channel, output_channel in zip(input_channels, self._channels):
            if not first_layer and input_channel != output_channel:
                self._layers.append(self._block(
                    input_channel, output_channel,
                    self._filter_height, self._filter_width, self._stride_height, self._stride_width
                ))
            else:
                self._layers.append(self._block(
                    input_channel, output_channel,
                    self._filter_height, self._filter_width, 1, 1
                ))
            first_layer = False

    def _setup(self, x, *calls, dropout=None):
        for layer in self._layers:
            x = layer(x, *calls, dropout=dropout, first=layer is self._layers[0])
        return x

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
