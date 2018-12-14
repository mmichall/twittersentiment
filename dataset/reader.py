from typing import List, Iterator, Union, Tuple
import pandas as pd


class CSVReader:

    def __init__(self, train_data_path: str, meta_path: str = None):
        self._train_data_path = train_data_path
        self._meta_path = meta_path

    def read(self, sents_cols: List[str], label_col: str, index_col: int = None, sep: str = '\t', merge_by: str = None) -> Iterator[Tuple[Union[str, List[str]], str]]:
        cols = sents_cols + [label_col]
        data = pd.read_csv(self._train_data_path, sep=sep, usecols=cols, index_col=index_col, header=0, encoding='utf8')
        if merge_by:
            data = data.reindex(columns=cols)
            data['sentence'] = data[sents_cols].apply(lambda x: ' <eou> '.join(x), axis=1)
            data = data[['sentence', 'label']]
        for row in data.itertuples(index=False):
            yield row