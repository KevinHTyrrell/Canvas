from experts.base_expert import BaseExpert


class ArgExpert(BaseExpert):
    # keeps a reference of all possible arguments and form #
    def _config(self):
        # depending on what form it takes it will get passed to an object that will parse properly #
        self._arg_table.update(
            {
                'type'                      : [str],
                'units'                     : [int],
                'activation'                : [str, tuple],
                'kernel_initializer'        : [str, tuple],
                'kernel_regularizer'        : [tuple],
                'dropout'                   : [float],
                'flatten_input'             : [bool],
                'flatten_output'            : [bool]
            }
        )

    def check_arg(self, arg_name, arg_val) -> None:
        arg_val_type = type(arg_val)
        val_type_allowed = self._arg_table.get(arg_name)
        # checks against arg_table #
        if val_type_allowed is None:
            raise AttributeError('{arg} is not an allowed hyperparameter'.format(arg=arg_name))
        if arg_val_type not in val_type_allowed:
            raise AttributeError('{arg} incorrect data type, must be {dt}'.format(arg=arg_name, dt=val_type_allowed))