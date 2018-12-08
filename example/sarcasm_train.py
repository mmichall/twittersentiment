import env
from ai.model import get_CNN_model
from dataset.reader import tweeter_normal_reader, tweeter_sarcasm_reader
from preprocessing.ekhprasis_libs import tweet_processor
import numpy as np

X = []
XX = []
Y = []
YY = []

maxlen = 0
for tweet in tweeter_normal_reader.read():
    sen = tweet_processor.pre_process_doc(tweet)
    if maxlen < len(sen):
        maxlen = len(sen)
    X.append(sen)
    Y.append([0, 1])

for tweet in tweeter_sarcasm_reader.read():
    sen = tweet_processor.pre_process_doc(tweet)
    if maxlen < len(sen):
        maxlen = len(sen)
    X.append(sen)
    Y.append([1, 0])


for x in X:
    XX.append(' '.join(x + (maxlen - len(x)) * ['_']))

model = get_CNN_model(maxlen)

model.fit(np.array(XX), np.array(Y), shuffle=True);

