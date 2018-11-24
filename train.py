import numpy as np
from tqdm import tqdm
import os
import random
import json

from dataset.reader import train_set_reader
from preprocessing.ekhprasis import tweet_processor
from ai.model import get_model
from ai.metrics import getMetrics

from embedding.deepmoji.deepmoji.sentence_tokenizer import SentenceTokenizer
from embedding.deepmoji.deepmoji.model_def import deepmoji_feature_encoding
from embedding.deepmoji.deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

from embedding.deep_moji import embed

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

NUM_FOLDS = 10

sentences = []
emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}
lab = np.eye(4)

for tuple in tqdm(train_set_reader.read()):
    sentences.append(' '.join(tweet_processor.pre_process_doc(tuple[0])))

maxlen = 0
for sentence in sentences:
    if maxlen < len(sentence):
        maxlen = len(sentence)

# DeepMoji embedding
deepmoji_x = embed(sentences, 6, maxlen)

x = sentences
y = []
for sentence in tqdm(sentences):
    y.append(lab[emotion2label[tuple[1].strip()]])

# Randomize data
combined = list(zip(x, deepmoji_x, y))
random.shuffle(combined)

x[:], deepmoji_x[:], y[:] = zip(*combined)

# Perform k-fold cross validation
metrics = {"accuracy": [],
           "microPrecision": [],
           "microRecall": [],
           "microF1": []}

print("Starting k-fold cross validation...")
for k in range(NUM_FOLDS):
    print('-' * 40)
    print("Fold %d/%d" % (k + 1, NUM_FOLDS))
    validationSize = int(len(x) / NUM_FOLDS)
    index1 = validationSize * k
    index2 = validationSize * (k + 1)

    xTrain = x[:index1] + x[index2:]
    xDeepmojiTrain = deepmoji_x[:index1] + deepmoji_x[index2:]
    yTrain = y[:index1] + y[index2:]
    xVal = x[index1:index2]
    xDeepmojiVal = deepmoji_x[index1:index2]
    yVal = y[index1:index2]

    model = get_model()
    model.fit([np.asarray(xTrain), np.asarray(xDeepmojiTrain)],
              np.asarray(yTrain),
              validation_data=([np.asarray(xVal), np.asarray(xDeepmojiVal)], np.asarray(yVal)),
              epochs=12,
              shuffle=True)

    predictions = model.predict(np.asarray(xVal), batch_size=32)
    accuracy, microPrecision, microRecall, microF1 = getMetrics(np.asarray(predictions), np.asarray(yVal))
    metrics["accuracy"].append(accuracy)
    metrics["microPrecision"].append(microPrecision)
    metrics["microRecall"].append(microRecall)
    metrics["microF1"].append(microF1)

print("\n============= Metrics =================")
print("Average Cross-Validation Accuracy : %.4f" % (sum(metrics["accuracy"]) / len(metrics["accuracy"])))
print("Average Cross-Validation Micro Precision : %.4f" % (
    sum(metrics["microPrecision"]) / len(metrics["microPrecision"])))
print("Average Cross-Validation Micro Recall : %.4f" % (sum(metrics["microRecall"]) / len(metrics["microRecall"])))
print("Average Cross-Validation Micro F1 : %.4f" % (sum(metrics["microF1"]) / len(metrics["microF1"])))

print("\n======================================")
