import env
import itertools
from dataset.dataset import FixedSplitDataSet
from dataset.reader import CSVReader
from preprocessor.ekhprasis import EkhprasisPreprocessor

labels_map = {'happy': 'sentiment',
              'sad': 'sentiment',
              'angry': 'sentiment',
              'others': 'nosentiment'}
labels = ['sentiment', 'nosentiment']

ekhprasis_preprocessor = EkhprasisPreprocessor(verbose=1)

sem_eval_dataset = FixedSplitDataSet(
    train_dataset=CSVReader(env.TRAIN_FILE_PATH, preprocessor=ekhprasis_preprocessor).read(
        sents_cols=['turn1', 'turn2', 'turn3'], label_col="label", merge_with=' eou '),
    test_dataset=CSVReader(env.DEV_FILE_PATH, preprocessor=ekhprasis_preprocessor).read(
        sents_cols=['turn1', 'turn2', 'turn3'], label_col="label", merge_with=' eou '),
    labels_map=labels_map
)


with open('train.csv', 'w', encoding='utf8') as f:
    for xy in itertools.zip_longest(sem_eval_dataset.iterate_train_x(), sem_eval_dataset.iterate_train_y()):
        f.write(str(labels.index(xy[1])) + ',' + xy[0] + '\n')


with open('val.csv', 'w', encoding='utf8') as f:
    for xy in itertools.zip_longest(sem_eval_dataset.iterate_test_x(), sem_eval_dataset.iterate_test_y()):
        f.write(str(labels.index(xy[1])) + ',' + xy[0] + '\n')