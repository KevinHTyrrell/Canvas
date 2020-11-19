from abc import ABC


class BaseMapper(ABC):
    def __init__(self):
        self._args = dict()