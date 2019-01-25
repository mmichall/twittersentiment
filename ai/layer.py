from keras import backend as K
from keras.layers import Layer


class WeightedConcatLayer(Layer):

    def __init__(self, output_dim, mask_input, **kwargs):
        self.output_dim = output_dim
        self._mask_input = mask_input
        super(WeightedConcatLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel_r = self.add_weight(name='kernel_r',
                                      shape=(input_shape[0], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        self.kernel_l = self.add_weight(name='kernel_l',
                                        shape=(input_shape[0], self.output_dim),
                                        initializer='uniform',
                                        trainable=True)
        super(WeightedConcatLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, mask=None):
        assert isinstance(x, list)
        a, b = x
        return K.dot(a, self.kernel_r) + K.dot(b, self.kernel_l)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim