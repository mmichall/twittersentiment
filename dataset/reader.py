import pandas as pd
import env
import json
from tqdm import tqdm
import csv

class DatasetReader:
    def __init__(self, path: str, mode: str):
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


class SarcasmReader:
    def __init__(self, comments_path: str, ids_path: str):
        self._comments_path = comments_path
        self._ids_path = ids_path

    def read(self):
        comments = {}
        with open(self._comments_path, 'r', encoding='utf8') as f:
            comments.update(json.loads(f.readline()))

        with open(self._ids_path) as fp, open('result.csv', 'w', encoding='utf8', newline='') as fres:
            writer = csv.writer(fres, delimiter='|')
            lines = fp.readlines()

            for line in tqdm(lines):
                row = []
                line = line.strip()

                parts = line.split('|')
                mains = parts[0].split(' ')
                responses = parts[1].split(' ')
                labels = parts[2].split(' ')

                _mains = []
                for main in mains:
                    if main in comments:
                        _mains.append(comments[main]['text'])

                topic = ' #main# '.join(_mains)

                _resps = []
                _ids = []
                for idx, main in enumerate(responses):
                    if main in comments:
                        _resps.append(comments[main]['text'])
                        _ids.append(labels[idx])


                responses = ' #resp# '.join(_resps)
                ids = ' '.join(_ids)

                row.append(topic)
                row.append(responses)
                row.append(ids)

                writer.writerow(row)


train_set_reader = DatasetReader(env.TRAIN_FILE_PATH, mode='train')
dev_set_reader = DatasetReader(env.DEV_FILE_PATH, mode='test')

sarcasm_set_reader = SarcasmReader(env.COMMENTS_FILE_PATH, env.IDS_SARC_FILE_PATH)
