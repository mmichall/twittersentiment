from keras import Sequential, Input, Model
from keras.engine import InputLayer
from keras.layers import Embedding, LSTM, Dropout, TimeDistributed, Activation, Dense
from keras_preprocessing.text import Tokenizer
from nltk.corpus import brown
from keras.utils import to_categorical
from keras_self_attention import SeqWeightedAttention
import numpy as np


def lstm(x_dim) -> Model:
    model = Sequential()
    model.add(InputLayer(input_shape=(x_dim, 1)))
    model.add(LSTM(256, return_sequences=True))
    model.add(SeqWeightedAttention())

    return model

X = []
Y = []
texts = []
categories = ['news', 'editorial', 'reviews']

tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts([' '.join(sentence) for sentence in brown.sents(categories=categories)])

for category in categories:
    x = []
    for sentence in brown.sents(categories=[category]):
        _tokens = tokenizer.texts_to_sequences(sentence[:10])
        if len(_tokens) < 10:
            _tokens = _tokens + [[0]] * (10 - len(_tokens))
        x.append(np.array(_tokens))
        if len(x) == 3:
            X.append(np.array(x))
            x = []
            Y.append(to_categorical(categories.index(category), num_classes=3))

print(np.array(X).shape)

input = Input(shape=(3, 10, 1))
layer = TimeDistributed(lstm(10))(input)
layer = LSTM(256, return_sequences=True)(layer)
layer = SeqWeightedAttention()(layer)
layer = Dense(3)(layer)
layer = Activation('softmax')(layer)

model = Model(input, layer)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])

model.fit(np.array(X),
          np.array(Y),
          epochs=10)
