import os.path

''' LOCAL '''
# DATASET_PATH = 'resources\\dataset'
# DEEPMOJI_WEIGHTS = 'resources\\dataset'
''' CLUSTER '''
DATASET_PATH = 'resources\\dataset'
DEEPMOJI_WEIGHTS = 'resources/deepmoji_weights.hdf5'
DEEPMOJI_VOCAB = 'resources/vocabulary.json'

TRAIN_FILE_NAME = 'train.txt'
DEV_FILE_NAME = 'dev.txt'
NORMAL_FILE_NAME = 'normal.txt'
IDS_FILE_NAME = 'ids.txt'
SARCASM_FILE_NAME = 'comments.json'
SARCASM_IDS_FILE_NAME = 'train-balanced.csv'
TRAIN_FILE_PATH = os.path.join(DATASET_PATH, TRAIN_FILE_NAME)
DEV_FILE_PATH = os.path.join(DATASET_PATH, DEV_FILE_NAME)
SARCASM_FILE_PATH = os.path.join('..', DATASET_PATH, 'sarcasm', SARCASM_FILE_NAME)
NORMAL_FILE_PATH = os.path.join('..', DATASET_PATH, 'sarcasm', NORMAL_FILE_NAME)
IDS_FILE_PATH = os.path.join('..', DATASET_PATH, 'sarcasm', IDS_FILE_NAME)

IDS_SARC_FILE_PATH = os.path.join('..', DATASET_PATH, 'SARC', SARCASM_IDS_FILE_NAME)
COMMENTS_FILE_PATH = os.path.join('..', DATASET_PATH, 'SARC', SARCASM_FILE_NAME)

USER1_SEP_START = ' <u1_start> '
USER2_SEP_START = ' <u2_start> '
USER1_SEP_STOP = ' <u1_stop> '
USER2_SEP_STOP = ' <u2_stop> '

''' CLUSTER '''
DATASET_PATH = '\data\\' + DATASET_PATH