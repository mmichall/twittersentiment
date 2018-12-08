import pandas as pd
import env
import json
from typing import List


class DatasetReader:
    def __init__(self, path, mode):
        if mode not in ['train', 'test']:
            raise AttributeError('Mode parameter can be train or test only.')
        self._path = path
        self._mode = mode

    def read(self):
        data = pd.read_csv(self._path, sep='\t', encoding='utf8', index_col='id')
        for tuple in data.itertuples():
            utterance = env.USER1_SEP_START + tuple.turn1 + env.USER1_SEP_STOP \
                        + env.USER2_SEP_START + tuple.turn2 + env.USER2_SEP_STOP \
                        + env.USER1_SEP_START + tuple.turn3 + env.USER1_SEP_STOP
            if self._mode == 'train':
                yield (utterance, tuple.label)
            else:
                yield (utterance)

class TweetSarcasReader:
    def __init__(self, path):
        self._path = path

    def read(self):
        with open(self._path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                yield line.split(',')[1].replace('#sarcasm','').replace('#Sarcasm','').replace('#SARCASM','').strip()

class SarcasmReader:
    def __init__(self, comments_paths, ids_path, ids_un_path):
        self._comments_paths = comments_paths
        self._ids_path = ids_path
        self._ids_un_path = ids_path

    def read(self):
        comments = {}
        for comment_path in self._comments_paths:
            with open(comment_path, 'r', encoding='utf8') as f:
                print(len(comments))
                comments.update(json.loads(''.join(f.readlines())))
        with open(self._ids_path) as fp, open(self._ids_un_path) as ufp:
            lines = fp.readlines() + ufp.readlines()

            for line in lines:
                parts = line.split('|')
                mains = parts[0].split(' ')
                responses = parts[1].split(' ')
                labels = parts[2].split(' ')

                for main in mains:
                    if main in comments:
                        print(comments[main]['text'])

                for idx, main in enumerate(responses):
                    if main in comments:
                        print(comments[main]['text'] + labels[idx] + '\n\n')


train_set_reader = DatasetReader(env.TRAIN_FILE_PATH, mode='train')
dev_set_reader = DatasetReader(env.DEV_FILE_PATH, mode='test')

#sarcasm_set_reader = SarcasmReader(env.SARCASM_FILE_PATH, env.SARCASM_IDS_FILE_PATH, env.SARCASM_IDS_UN_FILE_PATH)

