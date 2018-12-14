from abc import ABC, abstractmethod
import numpy as np
from typing import Iterator, Tuple, List, Union

from dataset.reader import CSVReader
import env


class DataSet(ABC):

    @abstractmethod
    def iterate_train(self): raise NotImplementedError

    @abstractmethod
    def iterate_train(self): raise NotImplementedError

    def __init__(self, name: str = None):
        self.name = name


class FixedSplitDataSet(DataSet):

    def __init__(self, train_generator: Iterator[Tuple[Union[str, List[str]], str]],
                 test_generator: Iterator[Tuple[Union[str, List[str]], str]],
                 name: str = None):
        super().__init__(name)
        self._train_generator = train_generator
        self._test_generator = test_generator

    def iterate_train(self) -> np.ndarray:
        for row in self._train_generator:
            yield row

    def iterate_test(self) -> np.ndarray:
        for row in self._test_generator:
            yield row


SemEvalDataSet = FixedSplitDataSet(train_generator=CSVReader(env.TRAIN_FILE_PATH).read(sents_cols=['turn1', 'turn2', 'turn3'], label_col="label", merge_by=' <eou> '),
                                   test_generator=CSVReader(env.DEV_FILE_PATH).read(sents_cols=['turn1', 'turn2', 'turn3'], label_col="label", merge_by=' <eou> '))
