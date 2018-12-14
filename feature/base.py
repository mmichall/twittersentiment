from abc import ABC, abstractmethod
from dataset.dataset import DataSet
from typing import Any, Callable, List
import numpy as np
from keras import Model, Input

class Feature(ABC):

    @abstractmethod
    def input(self): raise NotImplementedError

    @abstractmethod
    def transform(self, dataset: DataSet) -> np.ndarray: raise NotImplementedError

    @abstractmethod
    def name(self) -> str: raise NotImplementedError

    #TODO: max_len
    def _transform_by_func(self, dataset: DataSet, func: Callable[[str], Any], max_len: int, input3d: bool = False, input3d_dim: int = 1):
        shape: List = [dataset._len, max_len, input3d_dim]
        res = np.zeros(shape=shape)
        for sent_idx, sent in enumerate(dataset.iterate()):
            for word_idx, word in enumerate(sent[1].split()):
                if word_idx >= max_len: break
                print(word)
                feature_val = func(word)
                print(feature_val)
                if input3d:
                    res[sent_idx, word_idx, :] = feature_val
                else:
                    res[sent_idx, word_idx] = feature_val
        return res

    def _transform_by_func_to_flat_array(self, dataset: DataSet, func: Callable[[str], Any], max_len: int):
        res = []
        for sent_idx, sent in enumerate(dataset.iterate()):
            for word_idx, word in enumerate(sent):
                if word_idx >= max_len: break
                value: str = word[self.name()]
                feature_idx = func(value)
                res.append([feature_idx])
        return res