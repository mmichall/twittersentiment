from abc import abstractmethod
from typing import Union, List
from keras import Input
from keras.layers import Embedding, Lambda
import numpy as np
import pandas as pd
import unittest
import env

from dataset.dataset import DataSet
from feature.base import Feature


class WordFeature(Feature):

    def __init__(self, dataset: DataSet, name: str, embedding_vector_length: int):
        super().__init__(dataset, name=name)
        self._embedding_vector_length = embedding_vector_length
        self._casing_alphabet = ["numeric", "mostly_numeric", "contains_digit", "other"]
        with open(env.OFFENSIVE_WORDS_FILE_PATH) as of:
            self.offensive_words = set([word.lower().strip() for word in of.readlines()])
        with open(env.CONTROVERSIAL_WORDS_FILE_PATH) as of:
            self.controversial_words = set([word.lower().strip() for word in of.readlines()])

    def input_layer(self) -> Input:
        return Input(shape=(self._glove_vector_length,), name=self._name)

    def embedding_layer(self, trainable: bool = False) -> Union[Embedding, Lambda]:
        return Embedding(input_dim=len(self.word_2_index) + 1, output_dim=self._glove_vector_length,
                         weights=[self._word_embedding], trainable=trainable)

    def transform(self, **kwargs) -> np.ndarray:
        return super().transform(func=lambda x: self.convert_to_vector(x))

    def convert_to_vector(self, word: str) -> np.ndarray:
        special_chars = [1 if '!' in word else 0,
                         1 if '?' in word else 0]

        casing: np.ndarray = np.zeros(len(self._casing_alphabet), dtype=int)
        casing[self._casing_alphabet.index(self.get_casing(word))] = 1

        is_offensive = [self.is_offensive(word)]
        is_controvestial = [self.is_controversial(word)]

        res = []
        [res.extend(feature) for feature in [special_chars, casing, is_offensive, is_controvestial]]
        return res

    def get_casing(self, word: str):
   #     first_upper = len(word) > 0 and word[0].isupper()
        numeric = 0
     #   lower = 0
     #   upper = 0
        other = 0
        for char in word:
            if char.isnumeric():
                numeric += 1
            elif not char.isalnum():
                other += 1
       #     elif char.isupper():
       #         upper += 1
       #     elif char.islower():
       #         lower += 1
            else:
                other += 1
        if numeric == len(word):
            return "numeric"
        elif numeric >= len(word) / 2.0:
            return "mostly_numeric"
    #    elif upper == len(word):
    #        return "upper"
    #    elif lower == len(word):
    #        return "lower"
    #    elif first_upper and upper == 1:
    #        return "title"
        elif numeric > 0:
            return "contains_digit"
     #   elif upper > 0 and lower > 0:
     #       return "mixed"
        else:
            return "other"

    def is_offensive(self, word) -> int:
            return 1 if word.lower() in self.offensive_words else 0

    def is_controversial(self, word) -> int:
            return 1 if word.lower() in self.controversial_words else 0


class SpecialCharsFeatureTest(unittest.TestCase):

    def test_embed(self):
        _wordFeature = WordFeature(dataset=None, name='test', embedding_vector_length=100)

        print(_wordFeature.convert_to_vector("fuck"))
        print(_wordFeature.convert_to_vector("nazi"))

    if __name__ == '__main__':
        unittest.main()
