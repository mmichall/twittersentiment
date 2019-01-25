from typing import List, Union
from keras.layers import Dense, Bidirectional, Layer, LSTM, Concatenate, RNN, Input, SpatialDropout1D, Dropout, \
    TimeDistributed, Reshape, Activation, Flatten
from keras.models import Model
from keras_self_attention import SeqWeightedAttention as Attention
from feature.base import Feature
import keras.backend as K

# TODO: add logging results to file (experiment.record())

class RNNModelParams(object):
    def __init__(self, layers_size: Union[List[int], int], rnn_cell: RNN = LSTM,
                 spatial_dropout: Union[List[float], float] = 0., recurrent_dropout: Union[List[float], float] = 0.,
                 dropout_dense=0.,
                 dense_encoder_size: int = None,
                 class_weights: List[int] = None, bidirectional: bool = True):
        self.deep_lvl = len(layers_size)
        self.layers_size = layers_size
        self.rnn_cell = rnn_cell
        self.spatial_dropout = spatial_dropout
        self.dropout_dense = dropout_dense
        self.recurrent_dropout = recurrent_dropout
        self.dense_encoder_size = dense_encoder_size
        self.class_weights = class_weights
        self.bidirectional = bidirectional


class RNNModel:
    def __init__(self, inputs: Union[List[Input]], features: Union[List[Feature], Feature],
                 features_to_dense: Union[List[Feature], Feature], output: Dense,
                 params: RNNModelParams, attention: bool = True, _embed = None, input_direct = None):

        self._inputs = inputs
        self._input_direct = input_direct
        self._features = features
        self._features_to_dense = features_to_dense
        self._output = output
        self._attention = attention
        self._params = params
        self._embed = _embed

    def build(self):
        _embedding = [feature.embedding_layer(trainable=True) for idx, feature in enumerate(self._features)]
        _embedding_to_dense = [feature.embedding_layer(trainable=True) for idx, feature in enumerate(self._features_to_dense)]


        if len(_embedding) > 1:
            _layer = Concatenate()(_embedding)
        elif len(_embedding) == 1:
            _layer = _embedding[0]

        if len(_embedding_to_dense) > 1:
            _layer_to_dense = Concatenate()(_embedding_to_dense)
        elif len(_embedding_to_dense) == 1:
            _layer_to_dense = _embedding_to_dense[0]

        if self._input_direct is not None:
            _layer = self._input_direct
            print('self._input_direct is not None')

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
            _layer = Attention()(_layer)

        if len(_embedding_to_dense) != 0:
            _layer_to_dense = Reshape((6,))(_layer_to_dense)
            _layer = Concatenate()([_layer, _layer_to_dense])

        if self._params.dropout_dense:
            _layer = Dropout(self._params.dropout_dense)(_layer)
        if self._params.dense_encoder_size:
            _layer = Dense(self._params.dense_encoder_size, activation='relu')(_layer)

        if self._output:
            output = self._output(_layer)
        else:
            output = _layer

        self.__model = Model(inputs=self._inputs, outputs=output)

    def compile(self):
        self.__model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        self.__model.summary()

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, **kwargs):

        self.__model.fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle,
                         class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, **kwargs)

    def model(self):
        return self.__model


class HANModel:
    def __init__(self, inputs: Union[List[Input]], features: Union[List[Feature], Feature], output: Dense,
                 params: RNNModelParams, attention: bool = True):

        self._inputs = inputs
        self._features = features
        self._output = output
        self._attention = attention
        self._params = params

    def build(self):
        _embedding = [feature.embedding_layer(trainable=True) for idx, feature in enumerate(self._features)]

        if len(_embedding) > 1:
            _layer = Concatenate()(_embedding)
        else:
            _layer = _embedding[0]

        _layer = Reshape((-1, 3, 161))(_layer)
        for i, size in enumerate(self._params.layers_size):
            # if self._params.spatial_dropout[i] is not None:
            #     _layer = SpatialDropout1D(self._params.spatial_dropout[i])(_layer)
            hidden_layer = self._params.rnn_cell(size, recurrent_dropout=self._params.recurrent_dropout[i],
                                                 return_sequences=i != self._params.deep_lvl or self._attention)

            if self._params.bidirectional:
                _layer = TimeDistributed(Bidirectional(hidden_layer))(_layer)
            else:
                _layer = TimeDistributed(hidden_layer)(_layer)

        if self._attention:
            _layer = TimeDistributed(Attention())(_layer)

        _layer = Bidirectional(LSTM(300, recurrent_dropout=0.4, return_sequences=True))(_layer)
        _layer = Bidirectional(LSTM(300, recurrent_dropout=0.4, return_sequences=True))(_layer)
        _layer = Attention()(_layer)
        _layer = Capsule(num_capsule=2,
                         dim_capsule=8,
                         routings=3,
                         share_weights=True)(_layer)
        _layer = Flatten()(_layer)
        if self._params.dropout_dense:
            _layer = Dropout(self._params.dropout_dense)(_layer)
        if self._params.dense_encoder_size:
            _layer = Dense(self._params.dense_encoder_size, activation='relu')(_layer)

        output = self._output(_layer)

        self.__model = Model(inputs=self._inputs, outputs=output)

    def compile(self):
        self.__model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        self.__model.summary()

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, **kwargs):

        self.__model.fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle,
                         class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, **kwargs)

    def model(self):
        return self.__model


class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale