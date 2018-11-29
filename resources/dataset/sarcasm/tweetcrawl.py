import os
from twython import Twython
from twython.exceptions import TwythonError, TwythonRateLimitError
import env
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

normals = []
sarcasms = []

ids_set = set()

if os.path.exists('ids.txt'):
    with open('ids.txt', 'r') as f:
        ids_ = f.readlines()
        for id in ids_:
            if id:
                ids_set.add(int(id.strip()))

print(len(ids_set))

while(True):
    with open(env.NORMAL_FILE_PATH, 'r') as f:
        lines = f.readlines()
        for line in lines:
            print(line)
            try:
                if line.strip() in ids_set:
                    print("Omitted")
                    continue
                status = twitter.show_status(id=line.strip())
                print(status['text'])
                normals.append(status['text'])
            except TwythonRateLimitError as rle:
                print("Waiting 60 sec...")
                time.sleep(120)
            except TwythonError as e:
                print(str(e))

            with open('ids.txt', 'a', encoding='utf8') as fid:
                fid.write(line)
            if len(normals) % 10 == 0:
                with open('normals_tweetes.txt', 'a', encoding='utf8') as f:
                    for normal in normals:
                        f.write(line.strip() + ',' + normal.replace('\n', ' ') + '\n')

                normals = []

    with open(env.SARCASM_FILE_PATH, 'r') as f:
        lines = f.readlines()
        for line in lines:
            print(line)
            try:
                if line.strip() in ids_set:
                    print("Omitted")
                    continue
                status = twitter.show_status(id=line.strip())
                print(status['text'])
                sarcasms.append(status['text'])
            except TwythonRateLimitError as rle:
                print("Waiting 60 sec...")
                time.sleep(120)
            except TwythonError as e:
                print(str(e))

            with open('ids.txt', 'a', encoding='utf8') as fid:
                fid.write(line)
            if len(sarcasms) % 10 == 0:
                with open('sarcasm_tweetes.txt', 'a', encoding='utf8') as f:
                    for normal in sarcasms:
                        f.write(line.strip() + ',' + normal.replace('\n', ' ') + '\n')
                sarcasms = []
