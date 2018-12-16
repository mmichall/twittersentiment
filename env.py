import os.path

GLOVE_PATH = '../resources/glove'
DATASET_PATH = '../resources/dataset'
EXTERNAL_DIR_PATH = '../resources/external'
DEEPMOJI_WEIGHTS = '../resources/deepmoji_weights.hdf5'
DEEPMOJI_VOCAB = '../resources/vocabulary.json'

TRAIN_FILE_NAME = 'train.txt'
DEV_FILE_NAME = 'dev.txt'
GLOVE_FILE_NAME = 'glove.twitter.27B.25.txt'
OFFENSIVE_WORDS_FILE_NAME = 'offensive_words'
CONTROVERSIAL_WORDS_FILE_NAME = 'controversial_words'

TRAIN_FILE_PATH = os.path.join(DATASET_PATH, TRAIN_FILE_NAME)
DEV_FILE_PATH = os.path.join(DATASET_PATH, DEV_FILE_NAME)
GLOVE_FILE_PATH = os.path.join(GLOVE_PATH, GLOVE_FILE_NAME)
OFFENSIVE_WORDS_FILE_PATH = os.path.join(EXTERNAL_DIR_PATH, OFFENSIVE_WORDS_FILE_NAME)
CONTROVERSIAL_WORDS_FILE_PATH = os.path.join(EXTERNAL_DIR_PATH, CONTROVERSIAL_WORDS_FILE_NAME)

USER1_SEP_START = ' <u1_start> '
USER2_SEP_START = ' <u2_start> '
USER1_SEP_STOP = ' <u1_stop> '
USER2_SEP_STOP = ' <u2_stop> '