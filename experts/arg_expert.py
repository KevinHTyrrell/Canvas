from experts.base_expert import BaseExpert


class ArgExpert(BaseExpert):
    # keeps a reference of all possible arguments and form #
    def _config(self):
        # depending on what form it takes it will get passed to an object that will parse properly #
        param_dict = \
        {
            'activation'                : [str, tuple],
            'batch_norm'                : [bool],
            'dilution'                  : [tuple],
            'dropout'                   : [float],
            'filters'                   : [int],
            'flatten_input'             : [bool],
            'flatten_output'            : [bool],
            'kernel_initializer'        : [str, tuple],
            'kernel_regularizer'        : [tuple],
            'kernel_size'               : [int, tuple],
            'max_pool'                  : [int, tuple],
            'padding'                   : [str],
            'reshape'                   : [tuple],
            'strides'                   : [int, tuple],
            'type'                      : [str],
            'units'                     : [int],
            'upsample_input'            : [int, tuple],
            'upsample_output'           : [int, tuple]
        }
        self._args.update({'parameters': param_dict})
    def check_arg(self, arg_name, arg_val) -> None:
        arg_val_type = type(arg_val)
        val_type_allowed = self._args.get('parameters').get(arg_name)
        # checks against arg_table #
        if val_type_allowed is None:
            raise AttributeError('{arg} is not an allowed hyperparameter'.format(arg=arg_name))
        if arg_val_type not in val_type_allowed:
            raise AttributeError('{arg} incorrect data type, must be {dt}'.format(arg=arg_name, dt=val_type_allowed))