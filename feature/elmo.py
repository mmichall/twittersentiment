from abc import abstractmethod

from keras.layers import Embedding, Lambda, Layer

from feature.base import Feature
from dataset.dataset import DataSet
from typing import Any, Union
from keras import Model, Input
import numpy as np

import keras.backend as K
import tensorflow as tf
import tensorflow_hub as hub


class ELMoEmbeddingFeature(Feature):

    def __init__(self, name: str, max_len: int, input: Input):
        super().__init__(name=name, max_len=max_len, input=input)
        # self._elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

    def embedding_layer(self, trainable=False) -> Union[Embedding, Lambda, Layer]:
        return ElmoEmbeddingLayer()(self._input)

    def input_layer(self) -> Input:
        pass


class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))
        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(tf.squeeze(tf.cast(x, tf.string)),
                           as_dict=True,
                           signature='default')['elmo']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, 50 * ['_<PAD>_'])

    def compute_output_shape(self, input_shape):
        return None, 50, self.dimensions



