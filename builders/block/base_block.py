from abc import ABC
from tensorflow import keras
from mappers.arg_mapper import ArgMapper
from experts.layer_expert import LayerExpert


class BaseBlock(ABC):
    def __init__(self):
        self._args = dict()
        self._layers = list()
        self._arg_mapper = ArgMapper()
        self._layer_expert = LayerExpert()

    def build(self, input_layer: keras.layers.Layer):
        raise NotImplementedError

    def get_hyperparameters(self):
        return self._args.get('hyperparameters')

    def set_hyperparameters(self, args: dict):
        self._args.update({'hyperparameters': args})

    def get_output_layer(self):
        return self._args.get('output_layer')
