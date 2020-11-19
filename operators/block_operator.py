# WILL TAKE IN DICTIONARY OF PARSED ARGUMENTS AND PASS OUT ACCORDINGLY #
from operators.base_operator import BaseOperator
from experts.block_expert import BlockExpert


class BlockOperator(BaseOperator):
    def _config(self):
        self._expert = BlockExpert()
    def _set_transform_args(self, args: dict):
        self._args.get('blocks').get('transform').set_hyperparameters(args)
    def _set_train_args(self, args: dict):
        self._args.get('blocks').get('train').set_hyperparameters(args)
    def _set_transfer_args(self, args: dict):
        self._args.get('blocks').get('transfer').set_hyperparameters(args)

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

    def set_blocks(self, transform_layer, train_layer, transfer_layer):
        block_dict = {
            'transform'     : transform_layer,
            'train'         : train_layer,
            'transfer'      : transfer_layer,
        }
        self._args.update({'blocks': block_dict})
        
    def assign_hyperparameters(self, hyperparameters: dict):
        hp_dict = self._parse_hyperparameters(hyperparameters)
        assert len(hp_dict.keys()) == len(self._args.get('blocks'))
        for k, args in hp_dict.items():
            self._args['blocks'].get(k).set_hyperparameters(args)

    def build_blocks(self):
        block_dict = self._args.get('blocks')
        for block_name, block in block_dict.items():
            block.build()
