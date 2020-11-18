from builders.layer.base_layer import BaseBuilder
from tensorflow import keras


class DenseLayer(BaseBuilder):
    def _config(self):
        self._allowed_args = ['activation',
                              'dilution',
                              'flatten_input',
                              'output_shape',
                              'previous_layer',
                              'regularization',
                              'reshape_output',
                              'units']
        self._required_args = ['units']
        self._default_arg = dict()

    def _build_layer_skeleton(self, layer_config):
        layer_units = layer_config.get('units')
        layer_skeleton = keras.layers.Dense(units=layer_units)
        return layer_skeleton