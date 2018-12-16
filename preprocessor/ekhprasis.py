from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import Tokenizer
from preprocessor.ekhprasis_libs.dict.emoticons import emoticons
from preprocessor.ekhprasis_libs.dict.others import others
from preprocessor.base import Preprocessor
import logging

logging.getLogger().setLevel(logging.INFO)


class EkhprasisPreprocessor(Preprocessor):

    def __init__(self, verbose: int=0, omit=None,
                 normalize=None, annotate={"hashtag", "allcaps", "elongated", "repeated", 'emphasis'},
                 segmenter="twitter", corrector="twitter", unpack_hashtags=False, unpack_contractions=True,
                 spell_correct_elong=True, spell_correction=True, tokenizer=Tokenizer(lowercase=True),
                 dicts=None):
        super().__init__(name="EkhprasisPreprocessor", verbose=verbose)
        if dicts is None:
            dicts = [others, emoticons]
        if normalize is None:
            normalize = ['number']
        if omit is None:
            omit = ['email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date']
        logging.info("{} loading...".format(self._name))
        self.tweet_processor = TextPreProcessor(
            # omit terms
            omit=omit,
            # terms that will be normalized
            normalize=normalize,
            # terms that will be annotated
            annotate=annotate,

            # corpus from which the word statistics are going to be used
            # for word segmentation
            segmenter=segmenter,

            # corpus from which the word statistics are going to be used
            # for spell correction
            corrector=corrector,

            unpack_hashtags=unpack_hashtags,  # perform word segmentation on hashtags
            unpack_contractions=unpack_contractions,  # Unpack contractions (can't -> can not)
            spell_correct_elong=spell_correct_elong,  # spell correction for elongated words
            spell_correction=spell_correction,  # spell correction

            # select a tokenizer. You can use SocialTokenizer, or pass your own
            # the tokenizer, should take as input a string and return a list of tokens
            tokenizer=tokenizer.tokenize,

            # list of dictionaries, for replacing tokens extracted from the text,
            # with other expressions. You can pass more than one dictionaries.
            dicts=dicts
        )

    def _preprocess(self, sentence) -> str:
        return ' '.join(' '.join(self.tweet_processor.pre_process_doc(sentence)).split())
