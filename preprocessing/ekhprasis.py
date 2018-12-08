from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import Tokenizer
from preprocessing.ekhprasis_libs.dict.emoticons import emoticons
from preprocessing.ekhprasis_libs.dict.others import others

tweet_processor = TextPreProcessor(
    # omit terms
    omit=['email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date'],
    # terms that will be normalized
    normalize=['number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated", 'emphasis'},

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",

    unpack_hashtags=False,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words
    spell_correction=True, # spell correction

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=Tokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[others, emoticons]
)