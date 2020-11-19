from experts.base_expert import BaseExpert


class BlockExpert(BaseExpert):
    def _config(self):
        param_dict = {
            'activation'                : ('train', 'transfer'),
            'batch_norm'                : 'transfer',
            'dilution'                  : 'transfer',
            'dropout'                   : 'transfer',
            'filters'                   : 'train',
            'flatten_input'             : 'transform',
            'flatten_output'            : 'transfer',
            'kernel_initializer'        : 'train',
            'kernel_regularizer'        : 'train',
            'kernel_size'               : 'train',
            'max_pool'                  : 'transfer',
            'padding'                   : 'train',
            'reshape'                   : 'transfer',
            'strides'                   : 'train',
            'units'                     : 'train',
            'upsample_input'            : 'transform',
            'upsample_output'           : 'transfer',
            'type'                      : 'train'
        }
        self._args.update({'parameters': param_dict})
    def get_param_block(self, param_name: str):
        return self._args.get('parameters').get(param_name)

