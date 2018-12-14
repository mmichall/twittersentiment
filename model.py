from typing import List
from keras.layers import Input, LSTM, Concatenate
from keras import Model
from keras_self_attention import SeqWeightedAttention as Attention

#TODO: add logging results to file (experiment.record())

class RNNModelParams:
    def __init__(self,
                 hidden_layers: {},
                 spatial_dropout: List[float]=None,
                 recurrent_dropout: List[float]=None,
                 class_weights: List[int]=None):
        self.hidden_layers = hidden_layers
        self.dropout = spatial_dropout
        self.recurrent_dropout = recurrent_dropout
        self.class_weights = class_weights


class RNNModel:
    def __init__(self, inputs=None, outputs=None, attention: bool=True, params: RNNModelParams=RNNModelParams()):
        self.__inputs = inputs
        self.__outputs = outputs
        self.__attention = attention
        self.__params = params

    def build(self):
        self.__model = Model(inputs=self.__inputs, outpus=self.__outputs)

        if isinstance(self.__inputs, list):
            input = Concatenate()(self.__inputs, axis=1)
        else:
            input = self.__inputs

        for i, (rnn_type, size) in enumerate(self.__params.hidden_layers.items()):
            hidden_layer = rnn_type(size,
                                    recurrent_dropout=self.__params.recurrent_dropout[i],
                                    return_sequences=i != len(self.__params.hidden_layers) or self.__attention)(input)
            input = hidden_layer

        if self.__attention:
            attention = Attention()(input)
            input = attention

        out =





