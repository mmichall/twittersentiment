import env
from dataset.dataset import FixedSplitDataSet
from dataset.reader import CSVReader
from preprocessor.ekhprasis import EkhprasisPreprocessor, SimplePreprocessor
from keras.preprocessing.text import Tokenizer
import pandas as pd

ekhprasis_preprocessor = SimplePreprocessor(verbose=1)

# SemEvalDataSet = FixedSplitDataSet(
#     train_dataset=CSVReader(env.TRAIN_FILE_PATH_BL, preprocessor=ekhprasis_preprocessor).read(
#         sents_cols=None, label_col=None, merge_with=' <eou> '),
#     test_dataset=CSVReader(env.TRAIN_FILE_PATH_BL, preprocessor=ekhprasis_preprocessor).read(
#         sents_cols=None, label_col=None, merge_with=' <eou> '))

count = 0
data = pd.read_csv(env.TRAIN_FILE_PATH_BL, skip_blank_lines=False, header=None)
print(len(data))
