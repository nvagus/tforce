#!/usr/bin/env python3
# -- coding: utf8 --
# :author: nvagus
# :time: 2017/9/12-14:17
# :package: tforce.core

import contextlib
import threading

import numpy as np
import numpy.lib.format as nlf
import tensorflow as tf

from .base import Root
from .base import Widget


class Permutation(object):
    def __init__(self, size):
        self._size = size
        self._mutex = threading.Semaphore()
        self._iter = iter(self)

    def __iter__(self):
        def it():
            perm = np.arange(self._size)
            for i in range(self._size):
                r = np.random.randint(i, self._size)
                perm[i], perm[r] = perm[r], perm[i]
                yield perm[i]

        while True:
            yield from it()

    def next(self, size):
        with self._mutex:
            return np.hstack(next(self._iter) for _ in range(size))

    @property
    def size(self):
        return self._size


class Pool(Root, stderr='/dev/null', timeout=5):
    def __init__(self, workers, target):
        super(Pool, self).__init__()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._target = target
        self._workers = [threading.Thread(target=self.target) for _ in range(workers)]

    def target(self):
        while not self._stop.is_set():
            try:
                self._target()
            except Exception as e:
                try:
                    with self._lock, open(self.default.stderr, 'w') as file:
                        file.write(str(e))
                except KeyError:
                    pass

    def start(self):
        for worker in self._workers:
            worker.start()

    def stop(self):
        self._stop.set()
        for worker in self._workers:
            worker.join(timeout=self.default.timeout)

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class DataObj(dict):
    def __init__(self, data):
        super(DataObj, self).__init__(data)
        self._size = min((len(mat) for mat in self.values()))
        self._dtypes = [tf.as_dtype(mat.dtype) for mat in self.values()]
        self._shapes = [mat.shape[1:] for mat in self.values()]

    @property
    def size(self):
        return self._size

    @property
    def dtypes(self):
        return self._dtypes

    @property
    def shapes(self):
        return self._shapes


class AbstractDataSrc(object):
    def load(self, *keys, **kwargs):
        raise NotImplementedError()

    def save(self, *args, **kwargs):
        raise PermissionError('This data source is not modifiable')


class DataSrc(AbstractDataSrc):
    def __init__(self, **data):
        self._data = data

    def load(self, *keys):
        return DataObj({key: self._data[key] for key in keys})


class NpzDataSrc(AbstractDataSrc):
    def __init__(self, filename):
        self._filename = filename
        self._data = {}
        npz = np.load(filename)
        file = npz.zip.fp
        for key in npz.files:
            filename = '{}.npy'.format(key)
            npz.zip.open(filename)
            version = nlf.read_magic(file)
            shape, fortran_order, dtype = nlf.read_array_header_1_0(file) if version == (1, 0) \
                else nlf.read_array_header_2_0(file)
            self._data[key] = np.memmap(
                file, dtype=dtype, mode='r', shape=shape, order='F' if fortran_order else 'C', offset=file.tell())

    def load(self, *keys):
        return DataObj({key: self._data[key] for key in keys})

    def save(self, filename=None, **kwargs):
        filename = filename or self._filename
        self._data.update(kwargs)
        np.savez(filename, **self._data)

    @property
    def data(self):
        return self._data


class AbstractDataSet(object):
    def next(self, perm):
        raise NotImplementedError()

    @property
    def size(self):
        raise NotImplementedError()

    @property
    def keys(self):
        raise NotImplementedError()

    @property
    def dtypes(self):
        raise NotImplementedError()

    @property
    def shapes(self):
        raise NotImplementedError()


class DataSet(AbstractDataSet):
    def __init__(self, source: AbstractDataSrc, *keys):
        self._source = source
        self._data = source.load(*keys)
        self._size = self._data.size
        self._keys = list(self._data)
        self._dtypes = self._data.dtypes
        self._shapes = self._data.shapes

    def next(self, perm):
        return {key: self._data[key][perm] for key in self._keys}

    @property
    def source(self):
        return self._source

    @property
    def data(self):
        return self._data

    @property
    def size(self):
        return self._size

    @property
    def keys(self):
        return self._keys

    @property
    def dtypes(self):
        return self._dtypes

    @property
    def shapes(self):
        return self._shapes


class AbstractDataStream(Widget, capacity=10000, enqueue_size=100, workers=2):
    def __init__(self):
        super(AbstractDataStream, self).__init__()

    def using_workers(self, workers):
        raise NotImplementedError()

    @property
    def batch(self):
        raise NotImplementedError()

    @property
    def batch_size(self):
        raise NotImplementedError()

    def givens(self, batch_size):
        raise NotImplementedError()


class DataStream(AbstractDataStream):
    def __init__(self, data: AbstractDataSet, capacity=None, **kwargs):
        super(DataStream, self).__init__()
        self._data = data
        self._dtypes = self._data.dtypes
        self._shapes = self._data.shapes
        self._keys = self._data.keys
        self._capacity = capacity or self.default.capacity
        self._perm = Permutation(self._data.size)

    def _setup(self, sess):
        self._sess = sess
        self._queue = tf.PaddingFIFOQueue(
            capacity=self._capacity,
            dtypes=self._dtypes,
            shapes=self._shapes,
            names=self._keys
        )
        self._placeholders = {
            key: tf.placeholder(dtype, (None, *shape))
            for key, dtype, shape in zip(self._keys, self._dtypes, self._shapes)
        }
        self._batch_size = tf.placeholder(tf.int32, ())
        self._enqueue = self._queue.enqueue_many({key: self._placeholders[key] for key in self._keys})
        self._batch = self._queue.dequeue_many(self._batch_size)
        self._close = self._queue.close(cancel_pending_enqueues=True)

    def _process(self):
        perm = self._perm.next(self.default.enqueue_size)
        data = self._data.next(perm)
        self._sess.run(self._enqueue, feed_dict={self._placeholders[key]: data[key] for key in self._keys})

    def close(self):
        self._sess.run(self._close)

    @contextlib.contextmanager
    def using_workers(self, workers=None):
        workers = workers or self.default.workers
        with Pool(workers, self._process) as pool, contextlib.closing(self):
            yield pool

    @property
    def data(self):
        return self._data

    @property
    def queue(self):
        return self._queue

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def batch(self):
        return self._batch

    @property
    def keys(self):
        return self._keys

    @property
    def dtypes(self):
        return self._dtypes

    @property
    def shapes(self):
        return self._shapes

    def givens(self, batch_size):
        return {self._batch_size: batch_size}


class NpzDataStream(DataStream):
    def __init__(self, filename, *keys, capacity=None):
        data = DataSet(NpzDataSrc(filename), *keys)
        super(NpzDataStream, self).__init__(data, capacity)


class MultiDataStream(AbstractDataStream):
    def __init__(self, data: {str: AbstractDataSet}, capacity=None):
        super(MultiDataStream, self).__init__()
        self._subs = list(data)
        self._data = data
        self._capacity = capacity
        self._selected = self._subs[0]
        self._option_given = 0

    def _setup(self, sess):
        self._streams = [
            DataStream(self._data[sub], self._capacity, name=sub)
            for sub in self._data
        ]
        for stream in self._streams:
            stream.setup(sess)

        anyone = self._streams[0]
        self._keys = anyone.keys
        self._dtypes = anyone.dtypes
        self._shapes = anyone.shapes

        self._batch_size = tf.placeholder(tf.int32, ())
        self._option = tf.placeholder(tf.int32, ())
        self._queue = tf.PaddingFIFOQueue.from_list(
            self._option, [stream.queue for stream in self._streams]
        )

        self._batch = self._queue.dequeue_many(self._batch_size)

    @contextlib.contextmanager
    def using_workers(self, workers=None):
        with contextlib.ExitStack() as stack:
            for stream in self._streams:
                stack.enter_context(stream.using_workers(workers))
            yield stack

    @property
    def data(self):
        return self._data

    @property
    def queue(self):
        return self._queue

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def batch(self):
        return self._batch

    @property
    def option(self):
        return self._option

    @property
    def selected(self):
        return self._selected

    @selected.setter
    def selected(self, val):
        self._selected = val
        try:
            self._option_given = self._subs.index(val)
        except ValueError:
            raise ValueError(f'Invalid option: {val}')

    def givens(self, batch_size):
        return {
            self._batch_size: batch_size,
            self._option: self._option_given
        }


class MultiNpzDataStream(MultiDataStream):
    def __init__(self, data: {str: str}, *keys, capacity=None):
        data = {sub: DataSet(NpzDataSrc(data[sub]), *keys) for sub in data}
        super(MultiNpzDataStream, self).__init__(data, capacity)
