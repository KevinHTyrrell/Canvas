import re
from tensorflow import keras
from configs.keras_flags import KerasFlags
from mappers.base_mapper import BaseMapper


class DilutionMapper(BaseMapper):
    def __init__(self):
        self._dilution_dict = self._get_diluation_methods()
        self._config_mapper()

    @staticmethod
    def _get_diluation_methods() -> dict:
        dilution_dict = dict()
        for k, v in keras.layers.__dict__.items():
            if len(re.findall(KerasFlags.dilution_flags, k.lower())) > 0:
                dilution_dict.update({k.lower(): v})
        return dilution_dict

    def map(self, layer: dict) -> dict:
        dilution_elements = [element for element in list(layer.keys()) if element in list(self._dilution_dict.keys())]
        if len(dilution_elements) > 0:
            dilution_layer = self._dilution_dict.get(dilution_elements[0])
            dilution_amount = layer.get(dilution_elements[0])
            return {'type': 'layer', 'layer': dilution_layer(dilution_amount)}
        return self._default_arg
