import logging


class Preprocessor(object):

    def __init__(self, name: str = None, verbose: int = 0):
        self._name = name
        self._verbose = verbose
        self._sneak_look_counter = 0 if verbose == 0 else 10

    def preprocess(self, text) -> str:
        preprocessed = self._preprocess(text)
        self.sneak_look(text, preprocessed)
        return preprocessed

    def _preprocess(self, text) -> str: raise NotImplementedError

    def sneak_look(self, text, preprocessed):
        if self._verbose != 0 and self._sneak_look_counter != 0:
            logging.info('{}: {} ==>> {}'.format(self._name, text, preprocessed))
            self._sneak_look_counter = self._sneak_look_counter-1

