from typing import List

from keras.utils import to_categorical
import os
from keras_preprocessing.text import Tokenizer
from tqdm import tqdm
import numpy as np
import env
import tensorflow as tf
from dataset.dataset import FixedSplitDataSet, SimpleDataSet
from dataset.reader import CSVReader
from feature.base import Feature
from model import RNNModel, RNNModelParams
from keras import Input
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, GRU
import tensorflow as tf
from feature.pretrained import GensimPretrainedFeature
from feature.manual import WordFeature
from feature.elmo import ELMoEmbeddingFeature
from keras.layers import Dense
from metrics import MicroF1ModelCheckpoint

from preprocessor.ekhprasis import EkhprasisPreprocessor

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

MAX_LEN = 161
RNN_TYPE = LSTM
LAYERS_SIZE = [300, 300]
ATTENTION = True
SPATIAL_DROPOUT = [0.3, 0.5]
RECURRENT_DROPOUT = [0.3, 0.5]
DROPOUT_DENSE = 0.2
DENSE = 100
BATCH_SIZE = 32
EPOCHS = 25

labels_map = {'happy': 'sentiment',
              'sad': 'sentiment',
              'angry': 'sentiment',
              'others': 'nosentiment'}

ekhprasis_preprocessor = EkhprasisPreprocessor(verbose=1)

sem_eval_dataset_spc = FixedSplitDataSet(
    train_dataset=CSVReader(env.TRAIN_FILE_PATH, preprocessor=ekhprasis_preprocessor).read(
        sents_cols=['turn1', 'turn2', 'turn3'], label_col="label", merge_with=' <eou> '),
    test_dataset=CSVReader(env.DEV_FILE_PATH, preprocessor=ekhprasis_preprocessor).read(
        sents_cols=['turn1', 'turn2', 'turn3'], label_col="label", merge_with=' <eou> '),
    skip_labels=['others']
)

sem_eval_dataset_sent =  FixedSplitDataSet(
    train_dataset=CSVReader(env.TRAIN_FILE_PATH, preprocessor=ekhprasis_preprocessor).read(
        sents_cols=['turn1', 'turn2', 'turn3'], label_col="label", merge_with=' <eou> '),
    test_dataset=CSVReader(env.DEV_FILE_PATH, preprocessor=ekhprasis_preprocessor).read(
        sents_cols=['turn1', 'turn2', 'turn3'], label_col="label", merge_with=' <eou> ')
)

sem_eval_dataset_global =  FixedSplitDataSet(
    train_dataset=CSVReader(env.TRAIN_FILE_PATH, preprocessor=ekhprasis_preprocessor).read(
        sents_cols=['turn1', 'turn2', 'turn3'], label_col="label", merge_with=' <eou> '),
    test_dataset=CSVReader(env.DEV_FILE_PATH, preprocessor=ekhprasis_preprocessor).read(
        sents_cols=['turn1', 'turn2', 'turn3'], label_col="label", merge_with=' <eou> ')
)

sem_eval_dataset_sent.word2index = sem_eval_dataset_spc.word2index

input = Input(shape=(MAX_LEN,), name='one_hot_input')
text_input = Input(shape=(1,), dtype=tf.string, name='text_input')

manual_features = WordFeature(name='word_feature', input=input,
                              word2index=sem_eval_dataset_global.word2index,
                              embedding_vector_length=8, max_len=MAX_LEN)
glove_features = GensimPretrainedFeature(word2index=sem_eval_dataset_global.word2index, input=input,
                                         gensim_pretrained_embedding=env.W2V_310_FILE_PATH,
                                         embedding_vector_length=310, max_len=MAX_LEN)

manual_features_sent = WordFeature(name='word_feature', input=input,
                              word2index=sem_eval_dataset_sent.word2index,
                              embedding_vector_length=8, max_len=MAX_LEN)
glove_features_sent = GensimPretrainedFeature(word2index=sem_eval_dataset_sent.word2index, input=input,
                                         gensim_pretrained_embedding=env.W2V_310_FILE_PATH,
                                         embedding_vector_length=310, max_len=MAX_LEN)

elmo_features = ELMoEmbeddingFeature(name='ELMo_Embedding', max_len=MAX_LEN, input=text_input)

features: List[Feature] = [manual_features, glove_features, elmo_features]
features_sent: List[Feature] = [manual_features_sent, glove_features_sent, elmo_features]

params = RNNModelParams(layers_size=LAYERS_SIZE, spatial_dropout=SPATIAL_DROPOUT, recurrent_dropout=RECURRENT_DROPOUT,
                        dropout_dense=DROPOUT_DENSE,
                        dense_encoder_size=DENSE)

params_spc = RNNModelParams(layers_size=LAYERS_SIZE, spatial_dropout=[0.2, 0.5], recurrent_dropout=[0.2, 0.5],
                            dropout_dense=DROPOUT_DENSE,
                            dense_encoder_size=DENSE)

model = RNNModel(inputs=[input, text_input], features=features, output=Dense(2, activation='softmax', name="output"),
                 params=params,
                 attention=ATTENTION)

model_sent = RNNModel(inputs=[input, text_input], features=features_sent,
                     output=Dense(3, activation='softmax', name="output"),
                     params=params_spc,
                     attention=ATTENTION)

model.build()
model_sent.build()

model.model().load_weights("/data/03_F1_0.8953_catAcc_0.8921_LSTM_300X300_ATT_DROPOUT_0.3X0.5_100__.hdf5")
model_sent.model().load_weights(
    "/data/spc_11_F1_0.9605_catAcc_0.9520_trainable_LSTM_300X300_ATT_DROPOUT_0.2X0.5_100__.hdf5")
model.compile()
model_sent.compile()

X_val = [value for value in tqdm(sem_eval_dataset_global.iterate_test_x(max_len=MAX_LEN, one_hot=True))]
X_val_text = [value for value in tqdm(sem_eval_dataset_global.iterate_test_x(max_len=MAX_LEN, one_hot=False))]

X_val_spc = [value for value in tqdm(sem_eval_dataset_sent.iterate_test_x(max_len=MAX_LEN, one_hot=True))]
X_val_text_spc = [value for value in tqdm(sem_eval_dataset_sent.iterate_test_x(max_len=MAX_LEN, one_hot=False))]

Y = [y for y in tqdm(sem_eval_dataset_global.iterate_test_y(one_hot=True))]
Y_real = [y for y in tqdm(sem_eval_dataset_global.iterate_test_y(one_hot=False))]

val_data = ({'one_hot_input': np.array(X_val),
             'text_input': np.array(X_val_text)},
            {'output': np.array(Y)}
            )

label2emotion = {0: "sentiment", 1: "nosentiment"}
emotion2label = {"sentiment": 0, "nosentiment": 1}
spc_label2emotion = {0: "sad", 1: "happy", 2: "angry"}

global_predict = model.model().predict([np.array(X_val), np.array(X_val_text)], batch_size=BATCH_SIZE)
sent_predict = model_sent.model().predict([np.array(X_val_spc), np.array(X_val_text_spc)], batch_size=BATCH_SIZE)

discretePredictions = to_categorical(global_predict.argmax(axis=1), num_classes=2)
discretePredictions_spc = to_categorical(sent_predict.argmax(axis=1), num_classes=4)
# discretePredictions = np.around(val_predict)
global_pred_tmp = global_predict.argmax(axis=1)
sent_pred_tmp = sent_predict.argmax(axis=1)

val_predict = []
val_targ = []
lab2class = {"others": 0, 'happy': 1, 'sad': 2, 'angry': 3}
with open('/data/result/test.txt', "w", encoding="utf8") as fout:
    fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
    with open(env.DEV_FILE_PATH, encoding="utf8") as fin:
        fin.readline()
        for lineNum, line in enumerate(fin):
            if global_pred_tmp[lineNum] == 0:
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write(spc_label2emotion[sent_pred_tmp[lineNum]] + '\n')
                spc = spc_label2emotion[sent_pred_tmp[lineNum]]
            else:
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write('others' + '\n')
                spc = 'others'
            val_predict.append(lab2class[spc])
            val_targ.append(lab2class[Y_real[lineNum]])
        print("Completed. Model parameters: ")

discretePredictions = to_categorical(val_predict, num_classes=4)
val_targ = to_categorical(val_targ, num_classes=4)

truePositives = np.sum(discretePredictions * val_targ, axis=0)
falsePositives = np.sum(np.clip(discretePredictions - val_targ, 0, 1), axis=0)
falseNegatives = np.sum(np.clip(val_targ - discretePredictions, 0, 1), axis=0)

truePositives = truePositives[1:].sum()
falsePositives = falsePositives[1:].sum()
falseNegatives = falseNegatives[1:].sum()

microPrecision = truePositives / (truePositives + falsePositives)
microRecall = truePositives / (truePositives + falseNegatives)

# microF1 = (2 * microRecall * microPrecision) / (microPrecision + microRecall) if (microPrecision + microRecall) > 0 else 0
microF1 = 2 / (1.0 / microPrecision + 1.0 / microRecall)

print("- microF1: %f - true_positives: %d - false_positives %d - false_negatives: %d - precision: %f" % (
    microF1, int(truePositives), int(falsePositives), int(falseNegatives), microPrecision))
