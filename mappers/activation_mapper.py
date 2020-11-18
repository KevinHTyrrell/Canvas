from tensorflow import keras
import warnings
from mappers.base_mapper import BaseMapper


class ActivationMapper(BaseMapper):
    def __init__(self):
        self._advanced_activations = None
        self._standard_activations = None
        self._config_mapper()
        self._setup()

    def _setup(self):
        self._set_advanced_activations()
        self._set_standard_activations()

    def _set_advanced_activations(self) -> None:
        advanced_activation_dict = dict()
        layer_flag = KerasFlags.advanced_act_layer_flag
        for name, class_type in keras.layers.__dict__.items():
            if type(class_type) != type:
                continue
            if class_type.__dict__['__module__'] == layer_flag:
                advanced_activation_dict.update({name.lower(): class_type})
        self._advanced_activations =  advanced_activation_dict

    def _set_standard_activations(self) -> None:
        standard_activation_dict = dict()
        for name, function in keras.activations.__dict__.items():
            if not hasattr(function, '__call__'):
                continue
            standard_activation_dict.update({name.lower(): function})
        self._standard_activations = standard_activation_dict

    def map(self, layer: dict) -> dict:
        layer_activation = layer.get(KerasFlags.activation_flag)
        for act_name, act_function in self._standard_activations.items():
            if layer_activation == act_name:
                return {'type': 'standard', 'standard': act_function}
        for act_name, act_layer in self._advanced_activations.items():
            if layer_activation == act_name:
                layer_rate = layer.get(act_name)
                advanced_act_layer = act_layer(layer.get(layer_activation))
                if layer_rate is not None:
                    advanced_act_layer.__init__(layer_rate)
                    return {'type': 'advanced', 'advanced': advanced_act_layer}
                advanced_act_layer.__init__()
                return {'type': 'advanced', 'advanced': advanced_act_layer}
        warnings.warn('NOT A VALID ACTIVATION FUNCTION; RELU IS USED')
        return {'type': 'standard', 'standard': 'relu'}
