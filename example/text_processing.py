from preprocessing.ekhprasis import tweet_processor


sentences = [
'Mixed things  such as??	the things you do.	Have you seen minions??',
'Today I\'m very happy	and I\'m happy for you ❤	I will be marry',
'Woah bring me some	left it there oops	Brb',
'it is thooooo	I said soon master.	he is pressuring me',
'Wont u ask my age??	hey at least I age well!	Can u tell me how can we get closer??',
'I said yes	What if I told you I\'m not?	Go to hell	angry',
'Wheree I ll check	why tomorrow?	No I want now',
'Shall we meet	you say- you\'re leaving soon...anywhere you wanna go before you head?	?',
'Let\'s change the subject	I just did it .l.	You\'re broken',
'Your  pic  pz	thank you X‑D	wc	others',
'not mine	done for the day ?	can my meet to sexy girl',
'I want to play the game	if you just finished the game... then you haven\'t finished the game......	#Emojisong',
'Iam sory	why sorry ! 😿	I insult you',
'How much	depends on how long your internet has been out!!!	U have bf'
]

for sentence in sentences:
    print('%s  ->   %s' % (sentence, tweet_processor.pre_process_doc(sentence)))