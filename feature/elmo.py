from abc import abstractmethod

from keras.layers import Embedding, Lambda

from feature.base import Feature
from dataset.dataset import DataSet
from typing import Any, Union
from keras import Model, Input
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

class ELMoEmbeddingFeature(Feature):

    def __init__(self, name):
        self._name = name
        self._elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

    @abstractmethod
    def input_layer(self) -> Input: raise NotImplementedError

    @abstractmethod
    def embedding_layer(self, trainable=False) -> Union[Embedding, Lambda]: raise NotImplementedError

    def transform(self, dataset: DataSet) -> np.ndarray:
        yield self._elmo(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["elmo"]

    def name(self) -> str:
        return self._name