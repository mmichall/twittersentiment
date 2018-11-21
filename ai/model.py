import keras.layers as layers
from keras.models import Model
import tensorflow as tf

from embedding.elmo import elmo_model


def ElmoEmbedding(x):
    return elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["elmo"]


input_text = layers.Input(shape=(1,), dtype=tf.string)
embedding = layers.Lambda(ElmoEmbedding, output_shape=(1024,))(input_text)
lstm = layers.LSTM(100,  kernel_initializer='random_uniform')(embedding)
dense = layers.Dense(256, activation='relu')(lstm)
pred = layers.Dense(4, activation='softmax')(dense)

model = Model(inputs=input_text, outputs=pred)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
