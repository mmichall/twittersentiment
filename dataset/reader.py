from typing import Iterable
import pandas as pd
import env


class DatasetReader:
    def __init__(self, path: str, mode: str):
        if mode not in ['train', 'test']:
            raise AttributeError('Mode parameter can be train or test only.')
        self._path = path
        self._mode = mode

    def read(self) -> Iterable[str]:
        data = pd.read_csv(self._path, sep='\t', encoding='utf8', index_col='id')
        for tuple in data.itertuples():
            utterance = env.USER1_SEP_START + tuple.turn1 + env.USER1_SEP_STOP \
                      + env.USER2_SEP_START + tuple.turn2 + env.USER2_SEP_STOP \
                      + env.USER1_SEP_START + tuple.turn3 + env.USER1_SEP_STOP
            if self._mode == 'train':
                yield (utterance, tuple.label)
            else:
                yield (utterance)

train_set_reader = DatasetReader(env.TRAIN_FILE_PATH, mode='train')
dev_set_reader = DatasetReader(env.DEV_FILE_PATH, mode='test')