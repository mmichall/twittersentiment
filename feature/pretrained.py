from dataset.dataset import DataSet
from feature.base import OneHotFeature
from typing import Union
from keras import Input
from keras.layers import Lambda, Embedding
from tqdm import tqdm
import logging
import numpy as np
import gensim.downloader as api

logging.basicConfig(level=logging.INFO)


class GensimPretrainedFeature(OneHotFeature):

    def __init__(self, dataset: DataSet, gensim_pretrained_embedding: str, embedding_vector_length: int):
        super().__init__(dataset, name=gensim_pretrained_embedding)
        self._gensim_pretrained_glove = gensim_pretrained_embedding
        self._embedding_vector_length = embedding_vector_length
        self._word_embedding = np.zeros((len(self.word_2_index) + 1, self._embedding_vector_length))
        self.load_gensim_model()

    def input_layer(self) -> Input:
        return Input(shape=(self._embedding_vector_length,), name=self._name)

    def embedding_layer(self, trainable=False) -> Union[Embedding, Lambda]:
        return Embedding(input_dim=len(self.word_2_index) + 1, output_dim=self._embedding_vector_length,
                         weights=[self._word_embedding], trainable=trainable)

    def load_gensim_model(self):
        logging.info('Gensim {} is loading.'.format(self._gensim_pretrained_glove))
        self._gensim_embedding_model = api.load(self._gensim_pretrained_glove)
        logging.info('Creating {} embedding dictionary.'.format(self._gensim_pretrained_glove))
        for word, idx in tqdm(self.word_2_index.items()):
            if word in self._gensim_embedding_model.wv:
                self._word_embedding[idx] = self._gensim_embedding_model.wv.word_vec(word)