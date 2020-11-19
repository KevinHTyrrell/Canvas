from abc import ABC


class BaseOperator(ABC):
    def __init__(self):
        self._args = dict()
        self._config()

    def _config(self):
        raise NotImplementedError('NOT IMPLEMENTED')