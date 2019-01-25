from typing import List, Iterator, Union, Tuple
import pandas as pd
from preprocessor.base import Preprocessor
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

class CSVReader:

    def __init__(self, data_path: str, meta_path: str = None, preprocessor: Preprocessor=None, header: int=0):
        self._data_path = data_path
        self._meta_path = meta_path
        self._preprocessor = preprocessor
        self._header = header

    def read(self, sents_cols: List[str], label_col: str, index_col: int = None, sep: str = '\t', merge_with: str = None):
        cols=None
        if sents_cols:
            cols = sents_cols + [label_col]
        data = pd.read_csv(self._data_path, sep=sep, usecols=cols, index_col=index_col, header=self._header, encoding='utf8')
        if self._preprocessor:
            tqdm.pandas()
            logging.info('Preprocessing...')
            if sents_cols:
                data[sents_cols] = data[sents_cols].applymap(lambda x: self._preprocessor.preprocess(x))
                data_ = data[sents_cols]
            else:
                data_ = data.applymap(lambda x: self._preprocessor.preprocess(x))
        if merge_with:
            data = data.reindex(columns=cols)
            data['sentence'] = data_.apply(lambda x: merge_with.join(x), axis=1)
            if label_col:
                data = data[['sentence', 'label']]
            else:
                data['label'] = pd.Series()
                data = data[['sentence', 'label']]
        return data

