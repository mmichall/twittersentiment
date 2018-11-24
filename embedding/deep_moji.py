from __future__ import print_function, division
import json
import csv
import numpy as np
from embedding.deepmoji.deepmoji.sentence_tokenizer import SentenceTokenizer
from embedding.deepmoji.deepmoji.model_def import deepmoji_emojis
from embedding.deepmoji.deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH


def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

def embed(sentences, k, maxlen):

    print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)
    st = SentenceTokenizer(vocabulary, maxlen)
    tokenized, _, _ = st.tokenize_sentences(sentences)

    print('Loading model from {}.'.format(PRETRAINED_PATH))
    model = deepmoji_emojis(maxlen, PRETRAINED_PATH)
    model.summary()

    print('Running predictions.')
    prob = model.predict(tokenized)

    scores = []
    for i, t in enumerate(sentences):
        t_score = [0] * 64
        t_prob = prob[i]
        ind_top = top_elements(t_prob, k)
        for idx in ind_top:
            t_score[idx] = t_prob[idx]
        scores.append(t_score)

    return scores
