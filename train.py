import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset.reader import train_set_reader
from preprocessing.ekhprasis import tweet_processor
from ai.model import model

import tensorflow as tf
from keras import backend as K

with tf.Session() as sess:
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    x = []
    y = []
    emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}
    lab = np.eye(4)

    for tuple in tqdm(train_set_reader.read()):
        x.append(tweet_processor.pre_process_doc(tuple[0]))
        y.append(lab[emotion2label[tuple[1].strip()]])

    model.fit(np.array(x), np.array(y),
              batch_size=32,
              epochs=10,
              shuffle=True)
