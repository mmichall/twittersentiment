from dataset.reader import train_set_reader, dev_set_reader
from tqdm import tqdm
from preprocessing import ekhprasis

counts_ = {"others": 0, "happy": 0, "sad": 0, "angry": 0}


def class_counts():
    for row in train_set_reader.read():

        counts_[row[1].strip()] += 1


def print_words():
    words = {}
    for row in tqdm(dev_set_reader.read()):
        row = ekhprasis.tweet_processor.pre_process_doc(row)
        for word in row:
            word = word.lower()
            words.setdefault(word, 0)
            words[word] = words.get(word) + 1
    sorted_by_value = sorted(words.items(), key=lambda kv: kv[1], reverse=True)
    print(sorted_by_value)

print_words()
