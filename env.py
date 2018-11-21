import os.path

''' LOCAL '''
DATASET_PATH = 'resources\\dataset'
TRAIN_FILE_NAME = 'train.txt'
DEV_FILE_NAME = 'dev.txt'
TRAIN_FILE_PATH = os.path.join(DATASET_PATH, TRAIN_FILE_NAME)
DEV_FILE_PATH = os.path.join(DATASET_PATH, DEV_FILE_NAME)

USER1_SEP_START = ' <u1_start> '
USER2_SEP_START = ' <u2_start> '
USER1_SEP_STOP = ' <u1_stop> '
USER2_SEP_STOP = ' <u2_stop> '

''' CLUSTER '''
DATASET_PATH = ''