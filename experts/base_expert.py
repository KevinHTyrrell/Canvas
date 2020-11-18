from abc import ABC


class BaseExpert(ABC):
    def __init__(self):
        self._arg_table = dict()
        self._config()

    def _config(self):
        raise NotImplementedError('NOT IMPLEMENTED')