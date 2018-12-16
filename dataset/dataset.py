from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from preprocessor.base import Preprocessor

from dataset.reader import CSVReader
import env


class DataSet(ABC):

    @abstractmethod
    def iterate(self): raise NotImplementedError

    @abstractmethod
    def iterate_train(self): raise NotImplementedError

    @abstractmethod
    def iterate_test(self): raise NotImplementedError

    def __init__(self, name: str = None):
        self.name = name
        self.size = None


class FixedSplitDataSet(DataSet):

    def __init__(self, train_dataset: pd.DataFrame, test_dataset: pd.DataFrame, name: str = None):
        super().__init__(name)
        self._dataset = pd.concat([train_dataset, test_dataset], ignore_index=True)
        self.split_idx = len(train_dataset)
        self.size = self._dataset.size

    def iterate(self) -> np.ndarray:
        for row in self._dataset.itertuples():
            yield row

    def iterate_train(self) -> np.ndarray:
        for row in self._dataset.loc[:self.split_idx-1].itertuples():
            yield row

    def iterate_test(self) -> np.ndarray:
        for row in self._dataset.loc[self.split_idx:self.size].itertuples():
            yield row

    def len_(self):
        return
