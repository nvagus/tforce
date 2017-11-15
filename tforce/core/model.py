#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 2017/8/18-14:42
# :package: tforce.core

import contextlib
import inspect
import os
import subprocess

import tensorflow as tf

from .base import Scope
from .private import inspect_names as _inspect_names
from .private import make_iterable as _make_iterable
from .private import make_multiple as _make_multiple


class Slot(Scope, name='slot', max_output=10):
    """ A slot attaching to a model is a callable object that call the session to run tensors.
    """

    def __init__(
            self, model, frame,
            outputs=None, updates=None, givens=None, others=None,
            scalars=None, hists=None, images=None, audios=None,
            summaries=None, **__
    ):
        """ Initialize a slot, it is recommended to instantiate by model._add_slot method.
        :param model: the master model of the slot.
        :param frame: the frame that contains the variables of outputs, etc.
        :param outputs: output scalar tensors.
        :param updates: training tensors.
        :param givens: feed dict.
        :param others: others to output for debugging.
        :param scalars: summary scalars.
        :param hists: summary hists.
        :param images: summary images.
        :param audios: summary audios.
        """
        super(Slot, self).__init__()

        self._model = model
        self._local_step = 0

        self._outputs = _make_iterable(outputs)
        self._updates = _make_iterable(updates)
        self._givens = givens if givens else {}
        self._others = _make_iterable(others)

        self._outputs_count = len(self._outputs) + len(self._others)
        self._runnable = self._outputs + self._others + self._updates
        assert len(self._runnable), 'At least one output or update should be set'

        self._scalars = _make_iterable(scalars) if scalars else self._outputs
        self._images = _make_iterable(images)
        self._hists = _make_iterable(hists)
        self._audios = _make_iterable(audios)

        self._output_labels = _inspect_names(frame, self._outputs)
        self._scalar_labels = _inspect_names(frame, self._scalars)
        self._hist_labels = _inspect_names(frame, self._hists)
        self._image_labels = _inspect_names(frame, self._images)
        self._audio_labels = _inspect_names(frame, self._audios)

        self._summaries = _make_iterable(summaries)

    def _build(self):
        self._scalars_summary = tuple(tf.summary.scalar(name, x) for name, x in zip(self._scalar_labels, self._scalars))
        self._hists_summary = tuple(tf.summary.histogram(name, x) for name, x in zip(self._hist_labels, self._hists))
        self._images_summary = tuple(
            tf.summary.image(name, x, self.default.max_output) for name, x in zip(self._image_labels, self._images))
        self._audios_summary = tuple(
            tf.summary.audio(name, x, self.default.max_output) for name, x in zip(self._audio_labels, self._audios))
        assert self._scalars_summary or self._hists_summary or self._images_summary or self._audios_summary, \
            'At least one output or summary should be set'
        self._summaries = tf.summary.merge(
            self._scalars_summary + self._hists_summary + self._images_summary + self._audios_summary + self._summaries
        )

    def __call__(self, givens=None, valid_step=False):
        """ Run the outputs, updates, others, and summaries once.
        :param givens: updates for the feed dict.
        :param valid_step: whether to count the global or local step.
        :return: the session running result.
        """
        feed_dict = self.givens.copy()
        if givens is not None:
            feed_dict.update(givens)

        if not valid_step:
            self._local_step += 1
            self._model.global_step += 1

        self._model.next()

        if self._model.writer is not None:
            runnable, summaries = self._model.sess.run([self._runnable, self._summaries], feed_dict=feed_dict)
            self._model.writer.add_summary(summaries, global_step=self._model.global_step)
        else:
            runnable = self._model.sess.run(self._runnable, feed_dict=feed_dict)
        return _make_multiple(runnable[:self._outputs_count])

    @property
    def outputs(self):
        return _make_multiple(self._outputs)

    @property
    def updates(self):
        return _make_multiple(self._updates)

    @property
    def givens(self):
        return self._givens

    @property
    def labels(self):
        """ labels of output scalars.
        :return: labels of output scalars
        """
        return _make_multiple(self._output_labels)

    @property
    def local_step(self):
        return self._local_step

    @local_step.setter
    def local_step(self, step):
        self._local_step = step


class Model(
    Scope,
    name='model', gpu_options_allow_grouth=True, log_device_placement=False, allow_soft_placement=True,
    summary_dir='/tmp', tensorboard_port=6006
):
    """
    To build a model, you should inherit this class, and implement its protected build and setup method.
    The build method is not necessary to be implemented,
        but it is recommended that we build each part of the graph within build method
        and connect them by the setup method.
    """

    def __init__(self, **__):
        """ initialize a model
        """
        super(Model, self).__init__()
        self._graph = tf.Graph()
        self._sess = tf.Session(
            graph=self._graph,
            config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    allow_growth=self.default.gpu_options_allow_grouth
                ),
                log_device_placement=self.default.log_device_placement,
                allow_soft_placement=self.default.allow_soft_placement
            )
        )
        self._slots = {}
        self._global_step = 0
        self._writer = None
        self._streams = None
        self._data = None
        self._initializer = None

    def setup(self, *data_streams, **kwargs):
        """ Build the model with input data streams
        :param data_streams: the list of input data streams.
        """
        with self._sess.as_default(), self._graph.as_default():
            self._streams = data_streams
            for stream in self._streams:
                stream.setup()
            self._data = [stream.batch for stream in self._streams]
            with tf.variable_scope(self._name), tf.name_scope(self._scope):
                self._setup(*self._data, **kwargs)
                self._initializer = tf.variables_initializer(var_list=self.global_variables)
        self._sess.run(self._initializer)

    def _setup(self, *data_streams, **kwargs):
        """ Implement this method to use operators to build the graph.
        """
        raise NotImplementedError()

    __call__ = setup

    def next(self):
        """ Run the enqueue op for the model's buffers.
        """
        for stream in self._streams:
            stream.next()

    @contextlib.contextmanager
    def using_workers(self, workers=2):
        """ Use threads to run the enqueue ops for the data streams
        :param workers: how many threads working for each queue.
        :return: a context manager that runs the threads.
        """
        with contextlib.ExitStack() as stack:
            for stream in self._streams:
                stack.enter_context(stream.using_workers(workers))
            yield

    @contextlib.contextmanager
    def using_summaries(self, log_dir=None, port=None):
        """ Use summaries and open tensorboard.
        :param log_dir: the summary directory.
        :param port: the tensorboard port.
        :return: a context manager and the summary writer.
        """
        log_dir = log_dir or self.default.summary_dir
        port = port or self.default.tensorboard_port

        # with self._graph.as_default():
        #     for _, slot in self._slots.items():
        #         if not slot.built:
        #             slot.build()

        log_dir = os.path.join(log_dir, self._name)
        if os.path.exists(log_dir):
            for file in os.listdir(log_dir):
                os.remove(os.path.join(log_dir, file))
        else:
            os.mkdir(log_dir)
        self._writer = tf.summary.FileWriter(log_dir, graph=self._graph)
        with contextlib.closing(self._writer):
            server = subprocess.Popen(['python3', '-m', 'tensorboard.main', '--logdir', log_dir, str(port)])
            try:
                yield self._writer
            except Exception as e:
                server.send_signal(15)
                raise e
            finally:
                self._writer = None
        try:
            server.wait()
        except KeyboardInterrupt:
            server.send_signal(15)
        print('\nExit from tensorboard')

    def _add_slot(self, name, outputs=None, updates=None, givens=None, others=None,
                  scalars=None, hists=None, images=None, audios=None, summaries=None):
        """ Add a slot to the model.
        :param name: the name of the slot.
        :param outputs: output scalars.
        :param updates: assigning and training ops.
        :param givens: feed dict.
        :param others: for debugging.
        :param scalars: summary scalars, default is outputs.
        :param hists: summary hists.
        :param images: summary images.
        :param audios: summary audios.
        :param summaries: summaries from other parts of the graph, such as weight hists
        :return: a slot instance.
        """
        frame = inspect.currentframe().f_back
        slot = Slot(self, frame, outputs, updates, givens, others, scalars, hists, images, audios, summaries, name=name)
        self._slots[name] = slot
        return slot

    def save(self, filename, widget):
        with self._sess.as_default(), self._graph.as_default():
            widget.save(filename)

    def restore(self, filename, widget):
        with self._graph.as_default():
            self._sess.run(widget.restore(filename))

    @property
    def sess(self):
        return self._sess

    @property
    def writer(self):
        return self._writer

    @property
    def global_step(self):
        return self._global_step

    @global_step.setter
    def global_step(self, step):
        self._global_step = step

    @property
    def slots(self):
        return self._slots
