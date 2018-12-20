from abc import ABC, abstractmethod

from typing import Any, Callable, List, Tuple
import numpy as np
import pandas as pd
from keras_preprocessing.text import Tokenizer


class DataSet(ABC):
    @abstractmethod
    def iterate(self, max_len: int, one_hot: bool = False, ): raise NotImplementedError

    def __init__(self, name: str = None):
        self.name = name
        self.size = None
        self.word2index = None


class SimpleDataSet(DataSet):
    """
    TODO: balancing: oversampling, downsampling, smothe...
    """

    def __init__(self, dataset: pd.DataFrame, balancing=None):
        super().__init__()
        self._dataset = dataset
        self.size = self._dataset.size
        self.max_len = -1
        self._class_limit = None
        self._tokenizer: Tokenizer = Tokenizer(char_level=False, filters='')
        self._tokenizer.fit_on_texts([row for row in self.iterate()])
        self.word2index = self._tokenizer.word_index
        self._labels = list(set([row.label for row in self._dataset.itertuples()]))
        self._balancing = balancing

        self.class_count = {}
        for row in self._dataset.itertuples():
            _words_number = len(row.sentence.split(' '))
            if _words_number > self.max_len:
                self.max_len = _words_number

            self.class_count.setdefault(row.label, 0)
            self.class_count[row.label] += 1
        print('Classes counts: {}, max_len: {}'.format(str(self.class_count), str(self.max_len)))

        if balancing == 'downsample':
            self._class_limit = min(self.class_count.items(), key=lambda x: x[1])

    def _iterate(self, dataset: pd.DataFrame, max_len: int, one_hot: bool = False, shuffle=False):
        if shuffle:
            dataset = dataset.sample(frac=1).reset_index(drop=True)
        if self._class_limit:
            class_counter = {}
        for idx, row in enumerate(dataset.itertuples()):
            sentence = row.sentence
            if one_hot:
                if max_len:
                    one_hot_array = [0] * max_len
                    for idx_word, word in enumerate(row.sentence.split(' ')):
                        if idx_word >= max_len: continue
                        one_hot_array[idx_word] = self.word2index[word] if word in self.word2index else 0
                else:
                    one_hot_array = []
                    for word in row.sentence.split(' '):
                        one_hot_array.extend(self.word2index[word] if word in self.word2index else 0)
                sentence = one_hot_array
            if self._class_limit:
                class_counter.setdefault(row.label, 0)
                if class_counter[row.label] == self._class_limit[1]:
                    continue
                class_counter[row.label] += 1

            yield sentence

    def _iterate_y(self, dataset: pd.DataFrame, one_hot: bool = False) -> np.ndarray:
        for row in dataset.itertuples():
            if one_hot:
                _onehot = np.zeros(len(self._labels))
                _onehot[self._labels.index(row.label)] = 1
                yield _onehot
            else:
                yield row.label

    def iterate(self, max_len: int = None, one_hot: bool = False, shuffle=False):
        for item in self._iterate(self._dataset, max_len=max_len, one_hot=one_hot, shuffle=shuffle):
            yield item

    def iterate_x(self, max_len: int = None, one_hot: bool = False):
        return self._iterate(self._dataset, max_len=max_len, one_hot=one_hot)

    def iterate_y(self, one_hot: bool = False):
        return self._iterate_y(self._dataset, one_hot)


class FixedSplitDataSet(SimpleDataSet):
    def __init__(self, train_dataset: pd.DataFrame, test_dataset: pd.DataFrame):
        super().__init__(pd.concat([train_dataset, test_dataset], ignore_index=True))
        self.split_idx = len(train_dataset)

    def iterate_train_x(self, max_len: int = None, one_hot: bool = False):
        return self._iterate(self._dataset.loc[:self.split_idx - 1], max_len=max_len, one_hot=one_hot)

    def iterate_test_x(self, max_len: int = None, one_hot: bool = False):
        return self._iterate(self._dataset.loc[self.split_idx:self.size], max_len=max_len, one_hot=one_hot)

    def iterate_train_y(self, one_hot: bool = False):
        return self._iterate_y(self._dataset.loc[:self.split_idx - 1], one_hot)

    def iterate_test_y(self, one_hot: bool = False):
        return self._iterate_y(self._dataset.loc[self.split_idx:self.size], one_hot)
