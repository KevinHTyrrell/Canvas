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
        print(param_dict)
'''
activation
dropout
dilution
max_pool

# conflict so need to decide order of preference #
1. reshape
2. upsample_output
3. flatten_output
'''