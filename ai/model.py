import keras.layers as layers
from keras.models import Model
import tensorflow as tf

from embedding.elmo import elmo_model


def ElmoEmbedding(x):
    return elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["elmo"]


def get_model():
    input_text = layers.Input(shape=(1, ), dtype=tf.string, name="text_input")
    input_deepmoji = layers.Input(shape=(2304, ), name="deepmoji_input")

    embedding = layers.Lambda(ElmoEmbedding, output_shape=(None, 1024,))(input_text)
    sptDropout1D1 = layers.SpatialDropout1D(0.4)(embedding)
    lstm1 = layers.CuDNNLSTM(200, kernel_initializer='random_uniform', return_sequences=True)(sptDropout1D1)
    sptDropout1D2 = layers.SpatialDropout1D(0.4)(lstm1)
    lstm2 = layers.CuDNNLSTM(200, kernel_initializer='random_uniform', return_sequences=True)(sptDropout1D2)

    sentence, word_scores = Attention(return_attention=True)(lstm2)
    merged = layers.Concatenate()([input_deepmoji, sentence])

    dense2 = layers.Dense(300, activation='relu')(merged)
    pred = layers.Dense(4, activation='softmax', name="output")(dense2)

    model = Model(inputs=[input_text, input_deepmoji], outputs=pred)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    model.summary()

    return model
