from typing import List

from keras_preprocessing.text import Tokenizer
from tqdm import tqdm
import numpy as np
import env
from dataset.dataset import FixedSplitDataSet
from dataset.reader import CSVReader
from feature.base import Feature
from model import RNNModel, RNNModelParams
from keras import Input
from keras.callbacks import ModelCheckpoint
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

input=Input(shape=(10,))
#TODO: encapsulate embedding_vector_length argument
manual_features = WordFeature(name='word_feature', input=input,
                              word2index=sem_eval_dataset.word2index,
                              embedding_vector_length=8, max_len=10)

glove_features = GensimPretrainedFeature(word2index=sem_eval_dataset.word2index, input=input,
                                         gensim_pretrained_embedding='glove-twitter-25',
                                         embedding_vector_length=25, max_len=10)

features: List[Feature] = [glove_features, manual_features]

params = RNNModelParams(layers_size=[256, 256], spatial_dropout=[0.3, 0.3], recurrent_dropout=[0.3, 0.3],
                        dense_encoder_size=100)

model = RNNModel(inputs=input, features=features, output=Dense(4, activation='softmax', name="output"), params=params, attention=True)

model.build()
model.compile()

X_val = [np.array(value) for value in tqdm(sem_eval_dataset.iterate_test_x(max_len=10, one_hot=True))]
Y_val = [y for y in sem_eval_dataset.iterate_test_y(one_hot=True)]

filepath = "/data/LSTM_2_Att_Dense_100_cross_val_{epoch:02d}-{val_categorical_accuracy:.4f}.hdf5"
model.fit(x=np.array(X),
          y=np.array(Y),
          validation_data=(X_val, Y_val),
          batch_size=32,
          epochs=20,
          callbacks=[ModelCheckpoint(filepath=filepath, verbose=1, monitor='categorical_accuracy',
                                     save_weights_only=True, mode='auto')])


