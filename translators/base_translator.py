from abc import ABC


class BaseTranslator(ABC):
    def __init__(self):
        self._expert            = None
        self._config()

    def _config(self):
        raise NotImplementedError('NOT IMPLEMENTED')