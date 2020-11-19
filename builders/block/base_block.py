from abc import ABC


class BaseBlock(ABC):
    def __init__(self):
        self._args = dict()

    def _config(self):
        raise NotImplementedError