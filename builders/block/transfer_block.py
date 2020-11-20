from builders.block.base_block import BaseBlock, keras


class TransferBlock(BaseBlock):
    def build(self, input_layer: keras.layers.Layer):
        tensor_list = list()
        current_layer = input_layer
        param_dict = dict()
        hyperparameters = self._args.get('hyperparameters')
        for arg_name, arg_val in hyperparameters.items():
            arg_val_built = self._arg_mapper.map_arg(arg_name=arg_name, arg_val=arg_val)
            param_dict.update(arg_val_built)
        if param_dict.get('activation'):
            act_name, act_val = param_dict['activation'].values()
            activation_layer_skeleton = self._layer_expert.get_layer(act_name)
            activation_layer = activation_layer_skeleton(*act_val)(current_layer)
            current_layer = activation_layer
        if param_dict.get('dropout'):
            dropout_layer_skeleton = self._layer_expert.get_layer('dropout')
            dropout_layer = dropout_layer_skeleton(param_dict.get('dropout'))(current_layer)
            current_layer = dropout_layer
        if param_dict.get('dilution'):
            act_name, act_val = param_dict['dilution'].values()
            dilution_layer_skeleton = self._layer_expert.get_layer(act_name)
            dilution_layer = dilution_layer_skeleton(*act_val)(current_layer)
            current_layer = dilution_layer
        if param_dict.get('max_pool'):
            pool_dims = len(param_dict.get('max_pool'))
            pool_layer_name = 'maxpooling{n_dims}d'.format(n_dims=pool_dims)
            pool_layer_skeleton = self._layer_expert.get_layer(pool_layer_name)
            pool_layer = pool_layer_skeleton(param_dict.get('max_pool'))(current_layer)
            current_layer = pool_layer
        print(param_dict)



'''
    X    activation
    X    dropout
    X    dilution
    X    max_pool

# conflict so need to decide order of preference #
1. reshape
2. upsample_output
3. flatten_output
'''