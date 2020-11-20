from tensorflow import keras
from mappers.arg_mapper import ArgMapper
from builders.block.base_block import BaseBlock


class TrainBlock(BaseBlock):
    def build(self, input_layer: keras.layers.Layer):
        param_dict = dict()
        hyperparameters = self._args.get('hyperparameters')
        layer_type = hyperparameters['type'].get('val')
        del hyperparameters['type']
        layer_skeleton = self._layer_expert.get_layer(layer_type)
        for arg_name, arg_val in hyperparameters.items():
            arg_val_built = self._arg_mapper.map_arg(arg_name=arg_name, arg_val=arg_val)
            param_dict.update(arg_val_built)
        layer_built = layer_skeleton(**param_dict)
        self._args['layer'] = layer_built
        self._args['output_layer'] = layer_built
