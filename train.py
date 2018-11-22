import numpy as np
from tqdm import tqdm

from dataset.reader import train_set_reader
from preprocessing.ekhprasis import tweet_processor
from ai.model import model

import tensorflow as tf
from keras import backend as K

x = []
y = []
emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}
lab = np.eye(4)

for tuple in tqdm(train_set_reader.read()):
    x.append(' '.join(tweet_processor.pre_process_doc(tuple[0])))
    y.append(lab[emotion2label[tuple[1].strip()]])
with tf.Session() as sess:
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    for idx, xx in enumerate(x):
        model.fit(np.asarray(x), np.asarray(y),
                  epochs=10,
                  shuffle=True)
