from abc import ABC
from tensorflow import keras


class BaseBlock(ABC):
    def __init__(self):
        self._args = dict()

    def build(self, input_layer: keras.layers.Layer):
        raise NotImplementedError

    def set_hyperparameters(self, args: dict):
        self._args.update({'hyperparameters': args})