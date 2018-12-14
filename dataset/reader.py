from typing import List, Iterator, Union, Tuple
import pandas as pd
from preprocessing.base import Preprocessor

class CSVReader:

    def __init__(self, data_path: str, meta_path: str = None, preprocessor: Preprocessor=None):
        self._data_path = data_path
        self._meta_path = meta_path
        self._preprocessor = preprocessor

    def read(self, sents_cols: List[str], label_col: str, index_col: int = None, sep: str = '\t', merge_with: str = None):
        cols = sents_cols + [label_col]
        data = pd.read_csv(self._data_path, sep=sep, usecols=cols, index_col=index_col, header=0, encoding='utf8')
        if self._preprocessor:
            data[sents_cols] = data[sents_cols].applymap(lambda x: self._preprocessor.preprocess(x))
        if merge_with:
            data = data.reindex(columns=cols)
            data['sentence'] = data[sents_cols].apply(lambda x: merge_with.join(x), axis=1)
            data = data[['sentence', 'label']]
        return data