from abc import ABC, abstractmethod
from dataset.dataset import DataSet
from typing import Any, Callable
import numpy as np
from keras import Model, Input

class Feature(ABC):

    @abstractmethod
    def model(self, input: Any) -> Model: raise NotImplementedError

    @abstractmethod
    def input(self): raise NotImplementedError

    @abstractmethod
    def transform(self, dataset: DataSet) -> np.ndarray: raise NotImplementedError

    @abstractmethod
    def name(self) -> str: raise NotImplementedError

    def _transform_by_func(self, dataset: DataSet, func: Callable[[str], Any], input3d: bool = False,
                           input3d_dim: int = 1):

        res = []
        for sent in dataset.iterate_train():
            for word_idx, word in enumerate(sent):
                if word_idx >= dataset.sentence_length(): break
                value: str = word[self.name()]
                feature_val = func(value)
                if input3d:
                    res[sent_idx, word_idx, :] = feature_val
                else:
                    res[sent_idx, word_idx] = feature_val
        return res

    def _transform_by_func_to_flat_array(self, dataset: DataSet, func: Callable[[str], Any]):
        res = []
        for sent_idx, sent in enumerate(dataset.data):
            for word_idx, word in enumerate(sent):
                if word_idx >= dataset.sentence_length(): break
                value: str = word[self.name()]
                feature_idx = func(value)
                res.append([feature_idx])
        return res