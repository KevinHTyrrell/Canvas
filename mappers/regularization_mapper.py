import re
from tensorflow import keras
from configs.keras_flags import KerasFlags
from mappers.base_mapper import BaseMapper


class RegularizationMapper(BaseMapper):
    def __init__(self):
        self._regularizer_dict = self._get_regularizer_dict()
        self._config_mapper()

    @staticmethod
    def _get_regularizer_dict() -> dict:
        regularizer_dict = dict()
        for k, v in keras.regularizers.__dict__.items():
            if type(v) == type:
                regularizer_dict.update({k.lower(): v})
        return regularizer_dict

    def map(self, layer: dict) -> dict:
        reg_inputs = dict()
        for k, v in layer.items():
            if k.find(KerasFlags.reg_flag) != -1:
                reg_inputs.update({re.sub(KerasFlags.reg_flag + '|_', '', k): layer[k]})
        reg_values = tuple(element for element in reg_inputs.values())
        reg_str = ''.join(reg_inputs.keys())
        reg_function = self._regularizer_dict.get(reg_str, None)
        if reg_function is not None:
            return {'type': 'standard', 'standard': reg_function(*reg_values)}
        return self._default_arg


