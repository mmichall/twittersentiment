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
SARCASM_FILE_NAME1 = 'comments.json.002.002'
SARCASM_FILE_NAME2 = 'comments.json.002.003'
SARCASM_FILE_NAME3 = 'comments.json.012.009'
SARCASM_FILE_NAME4 = 'comments.json.012.010'
SARCASM_IDS_FILE_NAME = 'train-balanced.csv'
SARCASM_IDS_UN_FILE_NAME = 'train-unbalanced.csv'
TRAIN_FILE_PATH = os.path.join(DATASET_PATH, TRAIN_FILE_NAME)
DEV_FILE_PATH = os.path.join(DATASET_PATH, DEV_FILE_NAME)
SARCASM_FILE_PATH = [os.path.join('..', DATASET_PATH, 'sarcasm', SARCASM_FILE_NAME1), os.path.join('..', DATASET_PATH, 'sarcasm', SARCASM_FILE_NAME2), os.path.join('..', DATASET_PATH, 'sarcasm', SARCASM_FILE_NAME3), os.path.join('..', DATASET_PATH, 'sarcasm', SARCASM_FILE_NAME4)]
SARCASM_IDS_FILE_PATH = os.path.join('..', DATASET_PATH, 'sarcasm', SARCASM_IDS_FILE_NAME)
SARCASM_IDS_UN_FILE_PATH = os.path.join('..', DATASET_PATH, 'sarcasm', SARCASM_IDS_UN_FILE_NAME)

USER1_SEP_START = ' <u1_start> '
USER2_SEP_START = ' <u2_start> '
USER1_SEP_STOP = ' <u1_stop> '
USER2_SEP_STOP = ' <u2_stop> '


