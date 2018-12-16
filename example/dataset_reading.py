import env
from dataset.dataset import FixedSplitDataSet
from dataset.reader import CSVReader
from preprocessor.ekhprasis import EkhprasisPreprocessor
from keras.preprocessing.text import Tokenizer

ekhprasis_preprocessor = EkhprasisPreprocessor(verbose=1)

SemEvalDataSet = FixedSplitDataSet(
    train_dataset=CSVReader(env.TRAIN_FILE_PATH, preprocessor=ekhprasis_preprocessor).read(
        sents_cols=['turn1', 'turn2', 'turn3'], label_col="label", merge_with=' <eou> '),
    test_dataset=CSVReader(env.DEV_FILE_PATH, preprocessor=ekhprasis_preprocessor).read(
        sents_cols=['turn1', 'turn2', 'turn3'], label_col="label", merge_with=' <eou> '))

tokenizer = Tokenizer(oov_token='0', char_level=False, filters='')

lists = [row.sentence.split(' ') for row in SemEvalDataSet.iterate()]
tokenizer.fit_on_texts(lists)

for row in SemEvalDataSet.iterate():
    print(row.sentence)
    print(tokenizer.texts_to_sequences(row.sentence.split(' ')))
