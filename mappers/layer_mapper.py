import warnings
from tensorflow import keras
from configs.keras_flags import KerasFlags
from mappers.base_mapper import BaseMapper


class LayerMapper(BaseMapper):
    def __init__(self):
        self._layer_case_mappings = self._get_layer_case_mapping()
        self._config_mapper()

    @staticmethod
    def _get_layer_case_mapping() -> dict:
        layer_map = dict()
        for k, v in keras.layers.__dict__.items():
            if type(v) != type:
                continue
            if v.__dict__['__module__'].find(KerasFlags.layer_flag) != -1: # check if layer
                layer_map.update({k.lower(): k})
        return layer_map

    def map(self, layer_config: dict):
        layer_name = layer_config.get('type', 'dense').lower()
        if layer_name in self._layer_case_mappings:
            return getattr(keras.layers, self._layer_case_mappings.get(layer_name))
        return None