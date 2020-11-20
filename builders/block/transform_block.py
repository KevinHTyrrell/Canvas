from builders.block.base_block import BaseBlock, keras


class TransformBlock(BaseBlock):
    def build(self, input_layer: keras.layers.Layer):
        layers_built = list()
        current_layer = input_layer
        param_dict = dict()
        hyperparameters = self._args.get('hyperparameters')
        for arg_name, arg_val in hyperparameters.items():
            arg_val_built = self._arg_mapper.map_arg(arg_name=arg_name, arg_val=arg_val)
            param_dict.update(arg_val_built)
        if param_dict.get('flatten_input'):
            flatten_layer_skeleton = self._layer_expert.get_layer('flatten')
            flatten_layer = flatten_layer_skeleton()(current_layer)
            current_layer = flatten_layer
            layers_built.append(current_layer)
        if param_dict.get('upsample_input'):
            upsample_dims = len(param_dict.get('upsample_input'))
            upsample_layer_name = 'upsampling{n_dims}d'.format(n_dims=upsample_dims)
            upsample_layer_skeleton = self._layer_expert.get_layer(upsample_layer_name)
            upsample_layer = upsample_layer_skeleton(size=param_dict.get('upsample_input'))(current_layer)
            current_layer = upsample_layer
            layers_built.append(current_layer)
        self._args['output_layer'] = current_layer