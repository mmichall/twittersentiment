from typing import List

import os

import h5py
from keras.engine import InputLayer
from keras_preprocessing.text import Tokenizer
from tqdm import tqdm
import numpy as np
import env
import tensorflow as tf
from dataset.dataset import FixedSplitDataSet, SimpleDataSet
from dataset.reader import CSVReader
from feature.base import Feature
from model import RNNModel, RNNModelParams
from keras import Input, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional, Concatenate
import tensorflow as tf
from feature.pretrained import GensimPretrainedFeature
from feature.manual import WordFeature, SentenceFeature
from feature.elmo import ELMoEmbeddingFeature
from keras.layers import Dense
from metrics import MicroF1ModelCheckpoint

from preprocessor.ekhprasis import EkhprasisPreprocessor, SimplePreprocessor
from keras_self_attention import SeqWeightedAttention as Attention

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

MAX_LEN = 150
RNN_TYPE = LSTM
LAYERS_SIZE = [300, 300]
ATTENTION = True
SPATIAL_DROPOUT = [0.3, 0.5]
RECURRENT_DROPOUT = [0.3, 0.5]
DROPOUT_DENSE = None
DENSE = None
BATCH_SIZE = 32
EPOCHS = 25

labels_map = {'happy': 'sentiment',
              'sad': 'sentiment',
              'angry': 'sentiment',
              'others': 'nosentiment'}

ekhprasis_preprocessor = EkhprasisPreprocessor(verbose=1)
simple_preprocessor = SimplePreprocessor()

sem_eval_dataset_all = FixedSplitDataSet(
    train_dataset=CSVReader('/data/dataset/all.csv', preprocessor=simple_preprocessor, header=None).read(
        sents_cols=None, label_col=None, merge_with=' <eou> '),
    test_dataset=CSVReader(env.DEV_FILE_PATH_BL, preprocessor=simple_preprocessor, header=None).read(
        sents_cols=None, label_col=None, merge_with=' <eou> ')
)

sem_eval_dataset = FixedSplitDataSet(
    train_dataset=CSVReader(env.DEV_FILE_PATH_BL, preprocessor=simple_preprocessor, header=None).read(
        sents_cols=None, label_col=None, merge_with=' <eou> '),
    test_dataset=CSVReader(env.TEST_FILE_PATH_BL, preprocessor=simple_preprocessor, header=None).read(
        sents_cols=None, label_col=None, merge_with=' <eou> ')
)

for x in sem_eval_dataset_all.iterate_train_x():
    if len(x.split(' ')) > 50:
        print(x)

for x in sem_eval_dataset_all.iterate_test_x():
    if len(x.split(' ')) > 50:
        print(x)