import keras.layers as layers
from keras.models import Model
import tensorflow as tf

from embedding.elmo import elmo_model


def ElmoEmbedding(x):
    return elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["elmo"]


input_text = layers.Input(shape=(1,), dtype=tf.string)
embedding = layers.Lambda(ElmoEmbedding, output_shape=(None, 1024,))(input_text)
lstm = layers.LSTM(100, kernel_initializer='random_uniform')(embedding)
pred = layers.Dense(4, activation='softmax')(lstm)

model = Model(inputs=input_text, outputs=pred)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
