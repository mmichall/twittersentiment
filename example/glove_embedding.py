import env
from dataset.dataset import FixedSplitDataSet
from dataset.reader import CSVReader
from feature.glove import GensimPretrainedFeature
from preprocessor.ekhprasis import EkhprasisPreprocessor

ekhprasis_preprocessor = EkhprasisPreprocessor(verbose=1)

SemEvalDataSet = FixedSplitDataSet(
    train_dataset=CSVReader(env.TRAIN_FILE_PATH, preprocessor=ekhprasis_preprocessor).read(
        sents_cols=['turn1', 'turn2', 'turn3'], label_col="label", merge_with=' <eou> '),
    test_dataset=CSVReader(env.DEV_FILE_PATH, preprocessor=ekhprasis_preprocessor).read(
        sents_cols=['turn1', 'turn2', 'turn3'], label_col="label", merge_with=' <eou> '))

glove_100 = GensimPretrainedFeature(dataset=SemEvalDataSet, gensim_pretrained_embedding='glove-twitter-100', embedding_vector_length=100)

for embedding in glove_100.transform(max_len=100):
    print(embedding)