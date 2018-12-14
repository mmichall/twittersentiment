from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from preprocessing.ekhprasis import EkhprasisPreprocessor

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
        self._len = None
        self._max_len = None


class FixedSplitDataSet(DataSet):

    def __init__(self, train_dataset: pd.DataFrame, test_dataset: pd.DataFrame, name: str = None):
        super().__init__(name)
        self._dataset = pd.concat([train_dataset, test_dataset], ignore_index=True)
        self.split_idx = len(train_dataset)
        self._len = self._dataset.size

    def iterate(self) -> np.ndarray:
        for row in self._dataset.itertuples():
            yield row

    def iterate_train(self) -> np.ndarray:
        for row in self._dataset.loc[:self.split_idx-1].itertuples():
            yield row

    def iterate_test(self) -> np.ndarray:
        for row in self._dataset.loc[self.split_idx:self._len].itertuples():
            yield row

    def len_(self):
        return


SemEvalDataSet = FixedSplitDataSet(
    train_dataset=CSVReader(env.TRAIN_FILE_PATH, preprocessor=EkhprasisPreprocessor()).read(
        sents_cols=['turn1', 'turn2', 'turn3'], label_col="label", merge_with=' <eou> '),
    test_dataset=CSVReader(env.DEV_FILE_PATH, preprocessor=EkhprasisPreprocessor()).read(
        sents_cols=['turn1', 'turn2', 'turn3'], label_col="label", merge_with=' <eou> '))
