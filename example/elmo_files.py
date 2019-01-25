import env
from dataset.dataset import FixedSplitDataSet
from dataset.reader import CSVReader

sem_eval_dataset = FixedSplitDataSet(
    train_dataset=CSVReader(env.TRAIN_TAGS_FILE_PATH_, header=None).read(
        sents_cols=None, label_col=None, merge_with=' <eou> '),
    test_dataset=CSVReader(env.DEV_TAGS_FILE_PATH_, header=None).read(
        sents_cols=None, label_col=None, merge_with=' <eou> ')
)

dict_ = {}
for sentence in sem_eval_dataset.iterate_x():
    for word in sentence.split(' '):
        dict_.setdefault(word, 0)
        dict_[word] = dict_[word] + 1

sorted_by_value = sorted(dict_.items(), key=lambda kv: kv[1])