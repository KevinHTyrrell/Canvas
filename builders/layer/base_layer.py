from abc import ABC
from tensorflow import keras


class BaseBuilder(ABC):
    def __init__(self):
        self._allowed_args      = None
        self._input_shape       = None
        self._layer_skeleton    = None
        self._previous_layer    = None
        self._required_args     = None

    def _config(self):
        raise NotImplementedError("NOT IMPLEMENTED")

    def _build_layer_skeleton(self, layer_config):
        raise NotImplementedError("NOT IMPLEMENTED")