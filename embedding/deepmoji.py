from __future__ import print_function, division
import json
import numpy as np
from embedding.deepmoji_libs.deepmoji.sentence_tokenizer import SentenceTokenizer
from embedding.deepmoji_libs.deepmoji.model_def import deepmoji_feature_encoding
from embedding.deepmoji_libs.deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

def embed(sentences, maxlen):
    print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)
    st = SentenceTokenizer(vocabulary, maxlen)
    tokenized, _, _ = st.tokenize_sentences(sentences)

    print('Loading model from {}.'.format(PRETRAINED_PATH))
    model = deepmoji_feature_encoding(maxlen, PRETRAINED_PATH)
    model.summary()

    print('Running predictions.')
    prob = model.predict(tokenized)

    return prob