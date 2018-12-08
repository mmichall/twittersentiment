from nltk.corpus import brown
from texcla import experiment, data
from texcla.models import SentenceModelFactory, AttentionRNN
from texcla.preprocessing import SimpleTokenizer

X = []
y = []
categories = ['news', 'editorial', 'reviews']

for category in categories:
    for sentence in brown.sents(categories=[category]):
        X.append(' '.join(sentence))
        y.append(category)

tokenizer = SimpleTokenizer()

experiment.setup_data(X, y, tokenizer, 'data.bin', max_len=100)

ds = data.Dataset.load('data.bin')

factory = SentenceModelFactory(8, tokenizer.token_index, max_sents=500,
    max_tokens=200, embedding_type=None)
word_encoder_model = AttentionRNN()
sentence_encoder_model = AttentionRNN()

# Allows you to compose arbitrary word encoders followed by sentence encoder.
model = factory.build_model(word_encoder_model, sentence_encoder_model)

experiment.train(x=ds.X, y=ds.y, validation_split=0.1, model=model, word_encoder_model=word_encoder_model)