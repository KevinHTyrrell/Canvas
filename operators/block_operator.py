from tensorflow import keras
from operators.base_operator import BaseOperator
from experts.block_expert import BlockExpert


class BlockOperator(BaseOperator):
    def _config(self):
        self._expert = BlockExpert()

    def _parse_hyperparameters(self, hyperparameters: dict):
        param_dict = dict()
        for k, v in self._args.get('blocks').items():
            param_dict.update({k: dict()})
        for param_name, param_val in hyperparameters.items():
            param_block = self._expert.get_param_block(param_name)
            if type(param_block) is tuple and len(param_block) > 1:
                param_block = param_block[-1] if param_val.get('complex') else param_block[0]
            param_val.update({'block': param_block})
            param_dict[param_block].update({param_name: param_val})
        return param_dict

    def assign_hyperparameters(self, hyperparameters: dict):
        hp_dict = self._parse_hyperparameters(hyperparameters)
        assert len(hp_dict.keys()) == len(self._args.get('blocks'))
        for k, args in hp_dict.items():
            self._args['blocks'].get(k).set_hyperparameters(args)

    def build_blocks(self, input_layer: keras.layers.Layer):
        block_dict = self._args.get('blocks')
        tensor_list = list()
        tensor_list += [input_layer]
        for block_name, block in block_dict.items():
            tensor_list += block.build(input_layer)
            input_layer = block.get_output_layer()
        return tensor_list

    def set_blocks(self, transform_block, train_block, transfer_block):
        block_dict = {
            'transform'     : transform_block,
            'train'         : train_block,
            'transfer'      : transfer_block,
        }
        self._args.update({'blocks': block_dict})