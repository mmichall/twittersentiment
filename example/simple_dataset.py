from dataset.dataset import SimpleDataSet
from dataset.reader import CSVReader
from preprocessor.ekhprasis import EkhprasisPreprocessor
import env

ekhprasis_preprocessor = EkhprasisPreprocessor(verbose=1)

sem_eval_dataset = SimpleDataSet(dataset=CSVReader(env.TRAIN_FILE_PATH,
                                                   preprocessor=ekhprasis_preprocessor).read(
                                                   sents_cols=['turn1', 'turn2', 'turn3'],
                                                   label_col="label",
                                                   merge_with=' <eou> '),
                                 balancing='downsample',
                                 labels_map={'happy': 'sentiment',
                                             'sad': 'sentiment',
                                             'angry': 'sentiment',
                                             'others': 'nosentiment'})

print(len([row for row in sem_eval_dataset.iterate()]))
