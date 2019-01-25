from dataset.dataset import DataSet
from feature.base import OneHotFeature
from typing import Union, Dict, List
from keras import Input
from keras.layers import Lambda, Embedding
from tqdm import tqdm
import logging
import numpy as np
import os
import gensim.downloader as api
from gensim.models import KeyedVectors

logging.basicConfig(level=logging.INFO)


class GensimPretrainedFeature(OneHotFeature):

    def __init__(self, word2index: Dict, max_len: int, gensim_pretrained_embedding: str, embedding_vector_length: int, input: Input):
        super().__init__(name=gensim_pretrained_embedding, max_len=max_len, word2index=word2index, input= input)
        self._gensim_pretrained_glove = gensim_pretrained_embedding
        self._embedding_vector_length = embedding_vector_length
        self._word_embedding = np.zeros((len(self._word2index) + 1, self._embedding_vector_length))
        self.load_gensim_model()

    def embedding_layer(self, trainable=False) -> Union[Embedding, Lambda]:
        return Embedding(input_dim=self._word_embedding.shape[0], output_dim=self._word_embedding.shape[1],
                         weights=[self._word_embedding], trainable=False, mask_zero=True)(self._input)

    def load_gensim_model(self):
        logging.info('Gensim {} is loading.'.format(self._gensim_pretrained_glove))
        if os.path.isfile(self._gensim_pretrained_glove):
          self._gensim_embedding_model = KeyedVectors.load_word2vec_format (self._gensim_pretrained_glove, binary=False)
        else:
          self._gensim_embedding_model = api.load(self._gensim_pretrained_glove)
        logging.info('Creating {} embedding dictionary.'.format(self._gensim_pretrained_glove))
        for word, idx in tqdm(self._word2index.items()):
            if word in self._gensim_embedding_model.wv:
                self._word_embedding[idx] = self._gensim_embedding_model.wv.word_vec(word)