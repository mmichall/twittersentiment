from keras import Input, Sequential
import tensorflow as tf
from keras.layers import Lambda, Bidirectional, LSTM, Dense

import env
from ai.layer import WeightedConcatLayer
from dataset.dataset import SimpleDataSet
from dataset.reader import CSVReader
from feature.pretrained import GensimPretrainedFeature
from preprocessor.ekhprasis import EkhprasisPreprocessor

ekhprasis_preprocessor = EkhprasisPreprocessor(verbose=1)

sem_eval_dataset_dev = SimpleDataSet(dataset=CSVReader(env.DEV_FILE_PATH,
                                                       preprocessor=ekhprasis_preprocessor).read(
    sents_cols=['turn1', 'turn2', 'turn3'],
    label_col="label",
    merge_with=' <eou> '))


input = Input(shape=(161,), name='one_hot_input')

glove_features: GensimPretrainedFeature = GensimPretrainedFeature(word2index=sem_eval_dataset_dev.word2index,
                                                                  input=input,
                                                                  gensim_pretrained_embedding='glove-twitter-25',
                                                                  embedding_vector_length=25, max_len=161)

model = Sequential()

model.add(input)
model.add(glove_features.embedding_layer(trainable=False))
model.add(Bidirectional(LSTM(256, return_sequences=True), merge_mode=None))
model.add(WeightedConcatLayer(output_dim=(256, ), mask_input=None))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.summary()
