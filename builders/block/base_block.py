from abc import ABC


class BaseBlock(ABC):
    def __init__(self):
        self._args = dict()

    def _config(self):
        raise NotImplementedError

    def set_hyperparameters(self, args: dict):
        self._args.update({'hyperparameters': args})