from feature.base import Feature
from dataset.dataset import DataSet
import tensorflow as tf
import numpy as np
from keras import Input
import gensim.downloader as api

class SimpleFeature(Feature):

    def __init__(self, name):
        self._name = name
        self._glove_25_twitter = api.load("glove-twitter-25")

    def input(self):
        return Input(shape=(1,), dtype=tf.string, name=self.name() + '_Input')

    def transform(self, dataset: DataSet) -> np.ndarray:
        return self._transform_by_func(dataset, func=self.func, input3d_dim=25, max_len=100)

    def name(self) -> str:
        return self._name

    def func(self, word):
        if word in self._glove_25_twitter.wv:
            return self._glove_25_twitter.wv.word_vec(word)
        return [0] * 25
