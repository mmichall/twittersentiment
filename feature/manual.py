from abc import abstractmethod
from typing import Union, List, Dict
from keras import Input
from keras.layers import Embedding, Lambda
import numpy as np
import pandas as pd
import unittest

from tqdm import tqdm

import env
import logging
from dataset.dataset import DataSet
from feature.base import OneHotFeature

logging.basicConfig(level=logging.INFO)


class WordFeature(OneHotFeature):

    def __init__(self, name: str, word2index:Dict, embedding_vector_length: int, max_len: int, input:Input):
        super().__init__(name=name, word2index=word2index, max_len=max_len, input=input)
        self._embedding_vector_length = embedding_vector_length
        self._casing_alphabet = ["numeric", "mostly_numeric", "contains_digit", "other"]
        self._word_embedding = np.zeros((len(self._word2index) + 1, self._embedding_vector_length))
        with open(env.OFFENSIVE_WORDS_FILE_PATH) as of:
            self.offensive_words = set([word.lower().strip() for word in of.readlines()])
        with open(env.CONTROVERSIAL_WORDS_FILE_PATH) as of:
            self.controversial_words = set([word.lower().strip() for word in of.readlines()])
        with open(env.NEGATION_WORDS_FILE_PATH) as of:
            self.negation_words = set([word.lower().strip() for word in of.readlines()])

        self.create_weight_matrix()

    def embedding_layer(self, trainable: bool = False) -> Union[Embedding, Lambda]:
        return Embedding(input_dim=self._word_embedding.shape[0], output_dim=self._word_embedding.shape[1],
                         weights=[self._word_embedding], trainable=trainable, mask_zero=False)(self._input)

    def create_weight_matrix(self):
        logging.info('Creating {} embedding dictionary.'.format(self.name()))
        for word, idx in tqdm(self._word2index.items()):
            self._word_embedding[idx] = self.convert_to_vector(word)

    def convert_to_vector(self, word: str) -> np.ndarray:
        special_chars = [1 if '!' in word else 0,
                         1 if '?' in word else 0]

        is_offensive = [self.is_offensive(word)]
        is_controvestial = [self.is_controversial(word)]
        is_he_she = [self.is_he_she(word)]
        is_negation = [self.is_negation(word)]

        res = []
        [res.extend(feature) for feature in [special_chars, is_offensive, is_controvestial, is_negation, is_he_she]]
        return res

    def is_offensive(self, word) -> int:
            return 1 if word.lower() in self.offensive_words else 0

    def is_negation(self, word) -> int:
            return 1 if word.lower() in self.negation_words else 0

    def is_he_she(self, word) -> int:
            return 1 if word.lower() in ['he', 'she'] else 0

    def is_controversial(self, word) -> int:
            return 1 if word.lower() in self.controversial_words else 0


class SentenceFeature(OneHotFeature):
    def __init__(self, name: str, word2index: Dict, embedding_vector_length: int, max_len: int, input: Input):
        super().__init__(name=name, word2index=word2index, max_len=max_len, input=input)
        self._embedding_vector_length = embedding_vector_length
        self._word_embedding = np.zeros((len(self._word2index) + 1, self._embedding_vector_length))
        with open(env.OFFENSIVE_WORDS_FILE_PATH) as of:
            self.offensive_words = set([word.lower().strip() for word in of.readlines()])
        with open(env.CONTROVERSIAL_WORDS_FILE_PATH) as of:
            self.controversial_words = set([word.lower().strip() for word in of.readlines()])
        with open(env.NEGATION_WORDS_FILE_PATH) as of:
            self.negation_words = set([word.lower().strip() for word in of.readlines()])

        self.create_weight_matrix()

    def embedding_layer(self, trainable: bool = False) -> Union[Embedding, Lambda]:
        return Embedding(input_dim=self._word_embedding.shape[0], output_dim=self._embedding_vector_length,
                         weights=[self._word_embedding], trainable=trainable, mask_zero=False)(self._input)

    def create_weight_matrix(self):
        logging.info('Creating {} embedding dictionary.'.format(self.name()))
        for word, idx in tqdm(self._word2index.items()):
            self._word_embedding[idx] = self.convert_to_vector(word)

    def convert_to_vector(self, sentence: str) -> np.ndarray:
        special_chars = [1 if '!' in sentence else 0,
                         1 if '?' in sentence else 0]

        is_offensive = [self.is_offensive(sentence)]
        is_controvestial = [self.is_controversial(sentence)]
        is_he_she = [self.is_he_she(sentence)]
        is_negation = [self.is_negation(sentence)]

        res = []
        [res.extend(feature) for feature in [special_chars, is_offensive, is_controvestial, is_negation, is_he_she]]
        return res

    def is_offensive(self, sentence) -> int:
        for word in sentence.split(' '):
            if word.lower() in self.offensive_words:
                return 1
        return 0

    def is_negation(self, sentence) -> int:
        for word in sentence.split(' '):
            if word.lower() in self.negation_words:
                return 1
        return 0

    def is_he_she(self, sentence) -> int:
        for word in sentence.split(' '):
            if word.lower() in ['he', 'she']:
                return 1
        return 0

    def is_controversial(self, sentence) -> int:
        for word in sentence.split(' '):
            if word.lower() in self.controversial_words:
                return 1
        return 0


class SpecialCharsFeatureTest(unittest.TestCase):

    def test_embed(self):
        _wordFeature = WordFeature(dataset=None, name='test', embedding_vector_length=100)

        print(_wordFeature.convert_to_vector("fuck"))
        print(_wordFeature.convert_to_vector("nazi"))

    if __name__ == '__main__':
        unittest.main()
