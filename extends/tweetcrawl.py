import os
from twython import Twython
from twython.exceptions import TwythonError, TwythonRateLimitError
from tqdm import tqdm
import sched, time

s = sched.scheduler(time.time, time.sleep)
####input your credentials here
consumer_key = 'a26SUCCMkXpanedyQnyJ6qEhz'
consumer_secret = 's2CJKHryzzT3M778GP3uzYRfI9PBVfFj0hbX8DkGQsNZt33AOH'
access_token = '2885631262-2885631262-BNa5mGaZbHVvH4u6N9U8i5ixO20uMHeeYzbcuqB'
access_token_secret = 'JCfbVyZpekgdPeUw95DCPwYbdRkWSsUzMDas5fDPIyE0n'


twitter = Twython(
    consumer_key, consumer_secret, oauth_version=2)

ACCESS_TOKEN = twitter.obtain_access_token()

twitter = Twython(consumer_key, access_token=ACCESS_TOKEN)

ids_set = set()

if os.path.exists('ids.txt'):
    with open('ids.txt', 'r') as f:
        ids_ = f.readlines()
        for id in ids_:
            if id:
                ids_set.add(id.strip())

print(len(ids_set))


def download_tweets(file_with_ids, output_file):
    with open(file_with_ids, 'r') as f:
        idx = 0
        lines = f.readlines()
        while(True):
            line_ = lines[idx].strip()
            print(idx)
            try:
                if line_ in ids_set:
                    idx += 1
                    print("Omitted")
                    continue
                status = twitter.show_status(id=line_)
                idx += 1
                with open(output_file, 'a', encoding='utf8') as f:
                    f.write(line_ + ',' + status['text'].replace('\n', ' ') + '\n')
            except TwythonRateLimitError as rle:
                time.sleep(10)
            except TwythonError as e:
                idx += 1

            with open('ids.txt', 'a', encoding='utf8') as fid:
                fid.write(line_ + '\n')


download_tweets('../resources/dataset/sarcasm/normal.txt', '../resources/dataset/sarcasm/normal_tweets.txt')