from experts.base_expert import BaseExpert


class ArgExpert(BaseExpert):
    # keeps a reference of all possible arguments and form #
    def _config(self):
        # depending on what form it takes it will get passed to an object that will parse properly #
        param_dict = \
        {
            'type'                      : [str],
            'units'                     : [int],
            'activation'                : [str, tuple],
            'kernel_initializer'        : [str, tuple],
            'kernel_regularizer'        : [tuple],
            'dropout'                   : [float],
            'flatten_input'             : [bool],
            'flatten_output'            : [bool],

            'batch_norm'                : [bool],
            'reshape'                   : [tuple],
            'filters'                   : [int],
            'kernel_size'               : [int, tuple],
            'strides'                   : [int, tuple],
            'padding'                   : [str],
            'max_pool'                  : [int, tuple],
            'dilution'                  : [tuple],
            'upsample_input'            : [tuple]
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