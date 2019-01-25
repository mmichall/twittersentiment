import keras.layers as layers
from keras.models import Model
import tensorflow as tf
from keras_self_attention import SeqWeightedAttention as Attention
from embedding.elmo import ELMo


def lstm(seq_len: int):
    # input_deepmoji = layers.Input(shape=(2304, ), name="deepmoji_input")
    input_text = layers.Input(shape=(1,), dtype=tf.string, name="text_input")

    # embedding = layers.Embedding(168, 64)(input_text)
    embedding = layers.Lambda(ELMo, output_shape=(1024,))(input_text)

    spt_dropout_1 = layers.SpatialDropout1D(0.4)(embedding)
    lstm1 = layers.Bidirectional(
        layers.LSTM(350, kernel_initializer='random_uniform', return_sequences=True, recurrent_dropout=0.4))(
        spt_dropout_1)
    spt_dropout_2 = layers.SpatialDropout1D(0.3)(lstm1)
    lstm2 = layers.Bidirectional(
        layers.LSTM(350, kernel_initializer='random_uniform', return_sequences=True, recurrent_dropout=0.3))(
        spt_dropout_2)
    spt_dropout_3 = layers.SpatialDropout1D(0.2)(lstm2)
    lstm3 = layers.Bidirectional(
        layers.LSTM(300, kernel_initializer='random_uniform', return_sequences=True, recurrent_dropout=0.3))(
        spt_dropout_3)

    att = Attention()(lstm3)

    # merged = layers.Concatenate()([input_deepmoji, att])
    dense = layers.Dense(100, activation='relu')(att)
    pred = layers.Dense(2, activation='softmax', name="output")(dense)

    model = Model(inputs=input_text, outputs=pred)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    model.summary()

    return model

