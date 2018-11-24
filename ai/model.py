import keras.layers as layers
from keras.models import Model
import tensorflow as tf

from embedding.elmo import elmo_model


def ElmoEmbedding(x):
    return elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["elmo"]


def get_model():
    input_text = layers.Input(shape=(1, ), dtype=tf.string, name="text_input")
    input_deepmoji = layers.Input(shape=(64, ), name="deepmoji_input")

    embedding = layers.Lambda(ElmoEmbedding, output_shape=(None, 1024,))(input_text)
    sptDropout1D1 = layers.SpatialDropout1D(0.3)(embedding)
    lstm1 = layers.LSTM(150, kernel_initializer='random_uniform', return_sequences=True)(sptDropout1D1)
    sptDropout1D2 = layers.SpatialDropout1D(0.3)(lstm1)
    lstm2 = layers.LSTM(100, kernel_initializer='random_uniform', return_sequences=True)(sptDropout1D2)
    sptDropout1D3 = layers.SpatialDropout1D(0.2)(lstm2)
    lstm3 = layers.LSTM(50, kernel_initializer='random_uniform')(sptDropout1D3)
    dense1 = layers.Dense(1024, activation='relu')(lstm3)

   # merged = layers.Concatenate()([input_deepmoji, dense1])

   # dense2 = layers.Dense(64, activation='relu')(merged)
    pred = layers.Dense(4, activation='softmax', name="output")(dense1)

    model = Model(inputs=[input_text, input_deepmoji], outputs=pred)
  #  model = Model(inputs=[input_text], outputs=pred)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    model.summary()

    return model
