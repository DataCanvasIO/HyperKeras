# -*- coding:utf-8 -*-
"""

"""

import numpy as np
import tensorflow as tf

from hyperkeras import HyperKeras
from hyperkeras.search_space.cnn_search_space import cnn_search_space
from hyperkeras.search_space.dnn_search_space import dnn_search_space
from hypernets.core.callbacks import SummaryCallback
from hypernets.searchers.random_searcher import RandomSearcher


class Test_Dnn_Space():
    def test_dnn_space_hyper_model(self):
        rs = RandomSearcher(lambda: dnn_search_space(input_shape=10, output_units=2, output_activation='sigmoid'),
                            optimize_direction='max')
        hk = HyperKeras(rs, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'],
                        callbacks=[SummaryCallback()])

        x = np.random.randint(0, 10000, size=(100, 10))
        y = np.random.randint(0, 2, size=(100), dtype='int')

        hk.search(x, y, x, y, max_trials=3)

    def test_dnn_space(self):
        space = dnn_search_space(input_shape=10, output_units=2, output_activation='sigmod')
        space.random_sample()
        ids = []
        assert space.combinations

        def get_id(m):
            ids.append(m.id)
            return True

        space.traverse(get_id)
        assert ids

    def test_cnn_space_hyper_model(self):
        rs = RandomSearcher(
            lambda: cnn_search_space(input_shape=(28, 28, 1),
                                     output_units=10,
                                     output_activation='softmax',
                                     block_num_choices=[2, 3, 4, 5],
                                     filters_choices=[32, 64, 128],
                                     kernel_size_choices=[(1, 1), (3, 3)]),
            optimize_direction='max')
        hk = HyperKeras(rs, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'],
                        callbacks=[SummaryCallback()])

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Rescale the images from [0,255] to the [0.0,1.0] range.
        x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)
        print("Number of original training examples:", len(x_train))
        print("Number of original test examples:", len(x_test))

        # sample for speed up
        samples = 100
        hk.search(x_train[:samples], y_train[:samples], x_test[:int(samples / 10)], y_test[:int(samples / 10)],
                  max_trials=3, epochs=1)

    def test_cnn_space(self):
        space = cnn_search_space(input_shape=(50, 50), output_units=10, output_activation='softmax')
        space.random_sample()
        ids = []
        assert space.combinations

        def get_id(m):
            ids.append(m.id)
            return True

        space.traverse(get_id)
        assert ids
