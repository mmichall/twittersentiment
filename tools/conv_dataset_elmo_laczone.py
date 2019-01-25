import env
import pandas as pd
from tqdm import tqdm
from preprocessor.ekhprasis import EkhprasisPreprocessor

train_f_p = env.TRAIN_FILE_PATH
dev_f_p = env.DEV_FILE_PATH
test_f_p = env.TEST_FILE_PATH

train = pd.read_csv(train_f_p, sep='\t', usecols=['turn1', 'turn2', 'turn3'], header=0, encoding='utf8')
dev = pd.read_csv(dev_f_p, sep='\t', usecols=['turn1', 'turn2', 'turn3'], header=0, encoding='utf8')
test = pd.read_csv(test_f_p, sep='\t', usecols=['turn1', 'turn2', 'turn3'], header=0, encoding='utf8')

ekhp_processor = EkhprasisPreprocessor()

with open('train_bl_2.csv', 'w', encoding='utf8') as train_:
    for row in tqdm(train.itertuples()):
        train_.write(ekhp_processor.preprocess(str(row[1])) + ' ' + ekhp_processor.preprocess(str(row[2])) + ' ' + ekhp_processor.preprocess(str(row[3])) + '\n')
        train_.write(ekhp_processor.preprocess(str(row[2])) + ' ' + ekhp_processor.preprocess(str(row[3])) + '\n')
        train_.write(ekhp_processor.preprocess(str(row[3])) + '\n')

with open('dev_bl_2.csv', 'w', encoding='utf8') as dev_:
    for row in tqdm(dev.itertuples()):
        dev_.write(ekhp_processor.preprocess(str(row[1])) + ' ' + ekhp_processor.preprocess(str(row[2])) + ' ' + ekhp_processor.preprocess(str(row[3])) + '\n')
        dev_.write(ekhp_processor.preprocess(str(row[2])) + ' ' + ekhp_processor.preprocess(str(row[3])) + '\n')
        dev_.write(ekhp_processor.preprocess(str(row[3])) + '\n')

with open('test_bl_2.csv', 'w', encoding='utf8') as test_:
    for row in tqdm(test.itertuples()):
        test_.write(ekhp_processor.preprocess(str(row[1])) + ' ' + ekhp_processor.preprocess(str(row[2])) + ' ' + ekhp_processor.preprocess(str(row[3])) + '\n')
        test_.write(ekhp_processor.preprocess(str(row[2])) + ' ' + ekhp_processor.preprocess(str(row[3])) + '\n')
        test_.write(ekhp_processor.preprocess(str(row[3])) + '\n')