from abc import ABC, abstractmethod
from dataset.dataset import DataSet
from typing import Any, Callable, List, Union, Dict
import numpy as np
from keras import Input
from keras.layers import Embedding, Lambda
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm


class Feature(ABC):

    def __init__(self, name: str, max_len: int, input: Input):
        self._name = "{}_Input".format(name)
        self._max_len = max_len
        self._input = input

    def name(self) -> str:
        return self._name

    @abstractmethod
    def input_layer(self) -> Input:
        raise NotImplementedError

    @abstractmethod
    def embedding_layer(self, trainable=False) -> Union[Embedding, Lambda]:
        raise NotImplementedError


class OneHotFeature(Feature):

    def __init__(self, name: str, max_len: int, input: Input, word2index: Dict = None):
        super().__init__(name, max_len, input=input)
        self._word2index = word2index

    def input_layer(self) -> Input:
        return Input(shape=(self._max_len,), name=self._name)
