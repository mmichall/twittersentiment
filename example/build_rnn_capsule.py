from typing import List

import os
from keras_preprocessing.text import Tokenizer
from tqdm import tqdm
import numpy as np
import env
import tensorflow as tf
from dataset.dataset import FixedSplitDataSet, SimpleDataSet
from dataset.reader import CSVReader
from feature.base import Feature
from model import RNNModel, RNNModelParams
from keras import Input
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, GRU
import tensorflow as tf
from feature.pretrained import GensimPretrainedFeature
from feature.manual import WordFeature
from feature.elmo import ELMoEmbeddingFeature
from keras.layers import Dense
from metrics import MicroF1ModelCheckpoint

from preprocessor.ekhprasis import EkhprasisPreprocessor

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

MAX_LEN = 161
RNN_TYPE = LSTM
LAYERS_SIZE = [300, 300]
ATTENTION = True
SPATIAL_DROPOUT = [0.2, 0.5]
RECURRENT_DROPOUT = [0.2, 0.5]
DROPOUT_DENSE = 0.2
DENSE = 120
BATCH_SIZE = 64
EPOCHS = 25

labels_map = {'happy': 'sentiment',
              'sad': 'sentiment',
              'angry': 'sentiment',
              'others': 'nosentiment'}

ekhprasis_preprocessor = EkhprasisPreprocessor(verbose=1)

sem_eval_dataset = FixedSplitDataSet(
    train_dataset=CSVReader(env.TRAIN_FILE_PATH, preprocessor=ekhprasis_preprocessor).read(
        sents_cols=['turn1', 'turn2', 'turn3'], label_col="label", merge_with=' <eou> '),
    test_dataset=CSVReader(env.DEV_FILE_PATH, preprocessor=ekhprasis_preprocessor).read(
        sents_cols=['turn1', 'turn2', 'turn3'], label_col="label", merge_with=' <eou> '),
    labels_map=labels_map
)

sem_eval_dataset_dev = SimpleDataSet(dataset=CSVReader(env.DEV_FILE_PATH,
                                                   preprocessor=ekhprasis_preprocessor).read(
                                                   sents_cols=['turn1', 'turn2', 'turn3'],
                                                   label_col="label",
                                                   merge_with=' <eou> '),
                                 labels_map=labels_map,
                                 balancing='downsample')


sem_eval_dataset_dev.word2index = sem_eval_dataset.word2index

input = Input(shape=(MAX_LEN,), name='one_hot_input')
text_input = Input(shape=(1,), dtype=tf.string, name='text_input')
# TODO: encapsulate embedding_vector_length argument
manual_features = WordFeature(name='word_feature', input=input,
                              word2index=sem_eval_dataset.word2index,
                              embedding_vector_length=8, max_len=MAX_LEN)

# TODO: add trainable
glove_features = GensimPretrainedFeature(word2index=sem_eval_dataset.word2index, input=input,
                                         gensim_pretrained_embedding=env.W2V_310_FILE_PATH,
                                         embedding_vector_length=310, max_len=MAX_LEN)

elmo_features = ELMoEmbeddingFeature(name='ELMo_Embedding', max_len=MAX_LEN, input=text_input)

features: List[Feature] = [manual_features, glove_features, elmo_features]

params = RNNModelParams(layers_size=LAYERS_SIZE, spatial_dropout=SPATIAL_DROPOUT, recurrent_dropout=RECURRENT_DROPOUT,
                        dropout_dense=DROPOUT_DENSE,
                        dense_encoder_size=DENSE)

model = RNNModel(inputs=[input, text_input], features=features, output=Dense(3, activation='softmax', name="output"),
                 params=params,
                 attention=ATTENTION)

model.build()
model.compile()

X = [value for value in tqdm(sem_eval_dataset.iterate_train_x(max_len=MAX_LEN, one_hot=True))]
X_text = [value for value in tqdm(sem_eval_dataset.iterate_train_x(max_len=MAX_LEN, one_hot=False))]

X_val = [value for value in tqdm(sem_eval_dataset_dev.iterate_x(max_len=MAX_LEN, one_hot=True))]
X_val_text = [value for value in tqdm(sem_eval_dataset_dev.iterate_x(max_len=MAX_LEN, one_hot=False))]

Y_val = [y for y in tqdm(sem_eval_dataset_dev.iterate_y(one_hot=True))]
Y = [y for y in sem_eval_dataset.iterate_train_y(one_hot=True)]

val_data = ({'one_hot_input': np.array(X_val),
             'text_input': np.array(X_val_text)},
            {'output': np.array(Y_val)}
            )

filepath = "_{}_{}_{}_{}_{}_{}_".format(
    'LSTM' if RNN_TYPE == LSTM else 'GRU',
    'X'.join(str(x) for x in LAYERS_SIZE),
    'ATT' if ATTENTION else 'NO_ATT',
    'DROPOUT_' + 'X'.join(str(x) for x in SPATIAL_DROPOUT),
    str(DENSE) if DENSE != 0 else 'NO_DENSE',
    str() if DROPOUT_DENSE else '')
filepath = "/data/spc_{epoch:02d}_F1_{f1_micro_score:.4f}_catAcc_{val_categorical_accuracy:.4f}_trainable" + filepath + ".hdf5"

# model.model().predict((([np.array(X_val), np.array(X_val_text)], np.array(Y_val))[0]), batch_size=BATCH_SIZE)

model.fit(x=[np.array(X), np.array(X_text)],
          y=np.array(Y),
          validation_data=val_data,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          callbacks=[MicroF1ModelCheckpoint(filepath=filepath,
                                            verbose=1,
                                            monitor='categorical_accuracy',
                                            save_weights_only=True,
                                            mode='auto')])

predictions = []
# predictions = model.model().predict(np.array([np.array(X_val), np.array(X_val_text)]), batch_size=BATCH_SIZE)
predictions = predictions.argmax(axis=1)
label2emotion = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
with open('/data/result/test.txt', "w", encoding="utf8") as fout:
    fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
    with open(env.DEV_FILE_PATH, encoding="utf8") as fin:
        fin.readline()
        for lineNum, line in enumerate(fin):
            fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
            fout.write(label2emotion[predictions[lineNum]] + '\n')
        print("Completed. Model parameters: ")
