from typing import List, Union
from keras.layers import Dense, Bidirectional, Layer, LSTM, Concatenate, RNN, Input, SpatialDropout1D
from keras.models import Model
from keras_self_attention import SeqWeightedAttention as Attention
from feature.base import Feature

#TODO: add logging results to file (experiment.record())

class RNNModelParams(object):
    def __init__(self, layers_size: Union[List[int], int], rnn_cell: RNN = LSTM,
                 spatial_dropout: Union[List[float], float]=0., recurrent_dropout: Union[List[float], float]=0.,
                 dense_encoder_size: int = None, class_weights: List[int]=None, bidirectional:bool = True):
        self.deep_lvl = len(layers_size)
        self.layers_size = layers_size
        self.rnn_cell = rnn_cell
        self.spatial_dropout = spatial_dropout
        self.recurrent_dropout = recurrent_dropout
        self.dense_encoder_size = dense_encoder_size
        self.class_weights = class_weights
        self.bidirectional = bidirectional


class RNNModel:
    def __init__(self, features: Union[List[Feature], Feature], output: Dense,
                 params: RNNModelParams, attention: bool=True):

        self._features = features
        self._output = output
        self._attention = attention
        self._params = params

    def build(self):
        _inputs = [feature.input_layer() for feature in self._features]
        _embedding = [feature.embedding_layer()(_inputs[idx]) for idx, feature in enumerate(self._features)]

        if len(_embedding) > 1:
            _layer = Concatenate(_embedding)
        else:
            _layer = _embedding[0]

        for i, size in enumerate(self._params.layers_size):
            if self._params.spatial_dropout[i] is not None:
                _layer = SpatialDropout1D(self._params.spatial_dropout[i])(_layer)
            hidden_layer = self._params.rnn_cell(size, recurrent_dropout=self._params.recurrent_dropout[i],
                                                 return_sequences=i != self._params.deep_lvl or self._attention)
            if self._params.bidirectional:
                _layer = Bidirectional(hidden_layer)(_layer)
            else:
                _layer = hidden_layer(_layer)

        if self._attention:
            attention = Attention()(_layer)
            _layer = attention

        if self._params.dense_encoder_size:
            _layer = Dense(self._params.dense_encoder_size, activation='relu')(_layer)

        output = self._output(_layer)

        self.__model = Model(inputs=_inputs, outputs=output)

    def compile(self):
        self.__model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        self.__model.summary()



