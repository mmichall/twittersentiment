from abc import ABC, abstractmethod
from dataset.dataset import DataSet
from typing import Any, Callable, List, Union
import numpy as np
from keras import Input
from keras.layers import Embedding, Lambda
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm


class Feature(ABC):

    def __init__(self, dataset: DataSet, name: str):
        self._dataset = dataset
        self._name = "{}_Input".format(name)

    def name(self) -> str:
        return self._name

    @abstractmethod
    def input_layer(self) -> Input: raise NotImplementedError

    @abstractmethod
    def embedding_layer(self, trainable=False) -> Union[Embedding, Lambda]: raise NotImplementedError

    def transform(self, func: Callable[[str], Any], max_len: int, input3d: bool = False, input3d_dim: int = 1) -> np.ndarray:
        shape: List = [self._dataset.size, max_len, input3d_dim]
        res = np.zeros(shape=shape)
        for sent_idx, sent in tqdm(enumerate(self._dataset.iterate())):
            for word_idx, word in enumerate(sent[1].split()):
                if word_idx >= max_len: break
                feature_val = func(word)
                if input3d:
                    res[sent_idx, word_idx, :] = feature_val
                else:
                    res[sent_idx, word_idx] = feature_val
        return res


class OneHotFeature(Feature):

    def __init__(self, dataset: DataSet, name: str):
        super().__init__(dataset, name)
        self._tokenizer: Tokenizer = Tokenizer(char_level=False, filters='')
        self._tokenizer.fit_on_texts([row.sentence.split(' ') for row in dataset.iterate()])
        self.word_2_index = self._tokenizer.word_index

    def transform(self, max_len: int, **kwargs) -> np.ndarray:
        return super().transform(func=lambda x: self.word_2_index[x], max_len=max_len)
















