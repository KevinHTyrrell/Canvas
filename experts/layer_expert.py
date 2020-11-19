import numpy as np
from Levenshtein import distance as ld
from tensorflow import keras
from experts.base_expert import BaseExpert


class LayerExpert(BaseExpert):
    def _config(self):
        layer_dict = dict()
        for layer_name, layer_fn in keras.layers.__dict__.items():
            if layer_name.find('_') == -1:
                layer_dict.update({layer_name.lower(): layer_fn})
        self._args.update({'layers': layer_dict})

    def get_layer(self, layer_name: str):
        layer_name = layer_name.lower()
        assert layer_name in self._args.get('layers')
        layer_val = self._args['layers'].get(layer_name)
        return layer_val

    # don't implement yet #
    def _get_layer_fuzzy(self, layer_name: str):
        layer_name = layer_name.lower()
        layer_keys = list(self._args.get('layers').keys())
        layer_vals = list(self._args.get('layers').values())
        distances = [ld(layer_name.lower(), k) for k in self._args.get('layers').keys()]
        best_match_idx = np.argmin(distances)
        best_match_name = layer_keys[int(best_match_idx)]
        best_match_layer = layer_vals[int(best_match_idx)]
        return best_match_name, best_match_layer