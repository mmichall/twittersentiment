from keras_preprocessing.text import Tokenizer

import env
from dataset.dataset import FixedSplitDataSet
from dataset.reader import CSVReader
from model import RNNModel, RNNModelParams
from keras import Input
import tensorflow as tf
from feature.pretrained import GensimPretrainedFeature
from feature.manual import WordFeature
from keras.layers import Dense

from preprocessor.ekhprasis import EkhprasisPreprocessor

ekhprasis_preprocessor = EkhprasisPreprocessor(verbose=1)

sem_eval_dataset = FixedSplitDataSet(
    train_dataset=CSVReader(env.TRAIN_FILE_PATH, preprocessor=ekhprasis_preprocessor).read(
        sents_cols=['turn1', 'turn2', 'turn3'], label_col="label", merge_with=' <eou> '),
    test_dataset=CSVReader(env.DEV_FILE_PATH, preprocessor=ekhprasis_preprocessor).read(
        sents_cols=['turn1', 'turn2', 'turn3'], label_col="label", merge_with=' <eou> '))

glove_100 = GensimPretrainedFeature(dataset=sem_eval_dataset, gensim_pretrained_embedding='glove-twitter-100',
                                    embedding_vector_length=100)
#TODO: encapsulate embedding_vector_length argument
word_feature = WordFeature(name='word_feature', dataset=sem_eval_dataset, embedding_vector_length=8)

features = [glove_100]

params = RNNModelParams(layers_size=[256, 256], spatial_dropout=[0.3, 0.3], recurrent_dropout=[0.3, 0.3],
                        dense_encoder_size=100)

model = RNNModel(features=features, output=Dense(2, activation='softmax', name="output"), params=params, attention=True)

model.build()
model.compile()
