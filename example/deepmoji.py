from embedding.deepmoji_libs.deepmoji.sentence_tokenizer import SentenceTokenizer
from embedding.deepmoji_libs.deepmoji.model_def import deepmoji_feature_encoding
from embedding.deepmoji_libs.deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

import json

TEST_SENTENCES = [u'I love mom\'s cooking',
                  u'I love how you never reply back..',
                  u'I love cruising with my homies',
                  u'I love messing with yo mind!!',
                  u'I love you and now you\'re just gone..',
                  u'This is shit',
                  u'This is the shit']

maxlen = 30
batch_size = 32

print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)
st = SentenceTokenizer(vocabulary, maxlen)
tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)

print('Loading model from {}.'.format(PRETRAINED_PATH))
model = deepmoji_feature_encoding(maxlen, PRETRAINED_PATH)
model.summary()

print('Encoding texts..')
encoding = model.predict(tokenized)

print('First 5 dimensions for sentence: {}'.format(TEST_SENTENCES[1]))
print(len(encoding[1]))

# Now you could visualize the encodings to see differences,
# run a logistic regression classifier on top,
# or basically anything you'd like to do.