from abc import ABC, abstractmethod

from typing import Any, Callable, List, Tuple
import numpy as np
import pandas as pd
from keras_preprocessing.text import Tokenizer


class DataSet(ABC):

    @abstractmethod
    def iterate(self, max_len: int, one_hot:bool = False,): raise NotImplementedError

    @abstractmethod
    def iterate_train_x(self, max_len: int, one_hot:bool = False): raise NotImplementedError

    @abstractmethod
    def iterate_test_x(self, max_len: int, one_hot:bool = False): raise NotImplementedError

    @abstractmethod
    def iterate_train_y(self, one_hot: bool = False): raise NotImplementedError

    @abstractmethod
    def iterate_test_y(self, one_hot: bool = False): raise NotImplementedError


def __init__(self, name: str = None):
    self.name = name
    self.size = None
    self.word2index = None


class FixedSplitDataSet(DataSet):

    def __init__(self, train_dataset: pd.DataFrame, test_dataset: pd.DataFrame):
        super().__init__()
        self._dataset = pd.concat([train_dataset, test_dataset], ignore_index=True)
        self.split_idx = len(train_dataset)
        self.size = self._dataset.size
        self._tokenizer: Tokenizer = Tokenizer(char_level=False, filters='')
        self._tokenizer.fit_on_texts([row for row in self.iterate()])
        self.word2index = self._tokenizer.word_index
        self._labels = list(set([row.label for row in train_dataset.itertuples()]))
        print(self._labels)

    def __iterate(self, dataset: pd.DataFrame, max_len: int, one_hot: bool = False):
        for idx, row in enumerate(dataset.itertuples()):
            if not one_hot:
                yield row.sentence
            else:
                if max_len:
                    one_hot_array = [0] * max_len
                    for idx_word, word in enumerate(row.sentence.split(' ')):
                        if idx_word >= max_len: continue
                        one_hot_array[idx_word] = self.word2index[word] if word in self.word2index else 0
                else:
                    one_hot_array = []
                    for word in row.sentence.split(' '):
                        one_hot_array.extend(self.word2index[word] if word in self.word2index else 0)
                yield one_hot_array

    def __iterate_y(self, dataset: pd.DataFrame, one_hot: bool = False) -> np.ndarray:
        for row in dataset.itertuples():
            if one_hot:
                _onehot = np.zeros(len(self._labels))
                _onehot[self._labels.index(row.label)] = 1
                yield _onehot
            else:
                yield row.label

    def iterate(self, max_len: int = None, one_hot:bool = False):
        for item in self.__iterate(self._dataset, max_len=max_len, one_hot=one_hot):
            yield item

    def iterate_train_x(self, max_len: int=None, one_hot:bool = False):
        return self.__iterate(self._dataset.loc[:self.split_idx - 1], max_len=max_len, one_hot=one_hot)

    def iterate_test_x(self, max_len: int=None, one_hot:bool = False):
        return self.__iterate(self._dataset.loc[self.split_idx:self.size], max_len=max_len, one_hot=one_hot)

    def iterate_train_y(self, one_hot: bool = False):
        return self.__iterate_y(self._dataset.loc[:self.split_idx - 1], one_hot)

    def iterate_test_y(self, one_hot: bool = False):
        return self.__iterate_y(self._dataset.loc[self.split_idx:self.size], one_hot)
