import numpy as np
from tqdm import tqdm
import random
import env

from dataset.reader import train_set_reader, dev_set_reader
from preprocessor.ekhprasis_libs import tweet_processor
from ai.model import lstm
from ai.metrics import getMetrics

from embedding.deepmoji import embed

import keras.callbacks as callbacks

# PARAMS
NUM_FOLDS = 10
NUM_EPOCHS = 35
BATCH_SIZE = 32
CROSS_VALIDATION = True
NUM_CLASSES = 4

emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}
label2emotion = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
one_hot_lookup = np.eye(NUM_CLASSES)

x = []
y = []

for tuple in tqdm(train_set_reader.read()):
    xxx = tweet_processor.pre_process_doc(tuple[0])
    xxx = [xx.replace(' ', '_') for xx in xxx]
    x.append(xxx)
    y.append(one_hot_lookup[emotion2label[tuple[1].strip()]])

maxlen = 0
for i, sentence in enumerate(x):
    if maxlen < len(sentence):
        maxlen = len(sentence)

maxlen += 1
XX = []
for xx in x:
    XX.append(' '.join(xx + (maxlen - len(xx)) * ['_']))
x = XX

#DeepMoji embedding

'''if os.path.isfile(env.DEEPMOJI_FEATS_PCK):
  with open(env.DEEPMOJI_FEATS_PCK, 'rb') as df:
    deepmoji_x = pickle.load(df)
else:
  deepmoji_x = embed(x, 10, maxlen)
  with open(env.DEEPMOJI_FEATS_PCK, 'wb') as df:
    pickle.dump(deepmoji_x, df)

deepmoji_x = deepmoji_x.tolist()
'''

# Randomize data
combined = list(zip(x, y))
random.shuffle(combined)

x[:], y[:] = zip(*combined)

filepath = "/data/LSTM_3_Att_Dense_250_cross_val_{epoch:02d}-{val_categorical_accuracy:.4f}.hdf5"

if CROSS_VALIDATION:
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
        # xDeepmojiTrain = deepmoji_x[:index1] + deepmoji_x[index2:]
        yTrain = y[:index1] + y[index2:]
        xVal = x[index1:index2]
        #   xDeepmojiVal = deepmoji_x[index1:index2]
        yVal = y[index1:index2]

        model = lstm(maxlen)
        model.fit(np.asarray(xTrain),
                  np.asarray(yTrain),
                  validation_data=(np.asarray(xVal), np.asarray(yVal)),
                  epochs=NUM_EPOCHS,
                  shuffle=True,
                  callbacks=[callbacks.ModelCheckpoint(filepath=filepath,
                                                       verbose=1,
                                                       monitor='val_categorical_accuracy',
                                                       save_weights_only=True,
                                                       mode='auto')])

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

# Retraining model on entire data

print("Retraining model on entire data to create solution file")
model = lstm(maxlen)
model.fit(np.asarray(x), np.asarray(y),
          epochs=NUM_EPOCHS,
          batch_size=BATCH_SIZE,
          shuffle=True,
          callbacks=[callbacks.ModelCheckpoint(filepath=filepath,
                                               verbose=1,
                                               monitor='categorical_accuracy',
                                               save_weights_only=True,
                                               mode='auto')])

x = []
y = []
for tuple in tqdm(dev_set_reader.read()):
    x.append(' '.join(tweet_processor.pre_process_doc(tuple)))

deepmoji_x = embed(x, 10, maxlen)

maxlen = 0
for sentence in x:
    if maxlen < len(sentence.split()):
        maxlen = len(sentence)

model.save_weights('/data/elmo_deepmoji_GRU_weights.h5')
# model.load_weights('/data/baseline_elmo_2LSTM_Attention09-0.9463__.hdf5')

print("Creating solution file...")

predictions = model.predict([x, deepmoji_x], batch_size=BATCH_SIZE)
predictions = predictions.argmax(axis=1)

with open(env.SOLUTION_FILE_PATH, "w", encoding="utf8") as fout:
    fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
    with open(env.DEV_FILE_PATH, encoding="utf8") as fin:
        fin.readline()
        for lineNum, line in enumerate(fin):
            fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
            fout.write(label2emotion[predictions[lineNum]] + '\n')
        print("Completed. Model parameters: ")