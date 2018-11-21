from dataset.reader import DatasetReader
import env

train_set_reader = DatasetReader(env.TRAIN_FILE_PATH, mode='train')

for idx, line in enumerate(train_set_reader.read()):
    print('Line %d: %s' % (idx, line), end='\n')
