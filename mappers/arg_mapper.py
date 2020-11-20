from tensorflow import keras
from mappers.base_mapper import BaseMapper


class ArgMapper(BaseMapper):
    def _config(self):
        self._load_initializers()
        self._load_regularizers()

    def _load_initializers(self):
        init_dict = dict()
        for init_name, init_fn in keras.initializers.__dict__.items():
            if isinstance(init_fn, type):
                init_dict.update({init_name.lower(): init_fn})
        self._args['initializers'] = init_dict

    def _load_regularizers(self):
        reg_dict = dict()
        for init_name, init_fn in keras.regularizers.__dict__.items():
            if isinstance(init_fn, type):
                reg_dict.update({init_name.lower(): init_fn})
        self._args['regularizers'] = reg_dict

    def _get_initializer(self, init_name):
        return self._args['initializers'].get(init_name.lower())

    def _get_regularizer(self, reg_name):
        return self._args['regularizers'].get(reg_name.lower())

    def map_arg(self, arg_name: str, arg_val: dict):
        if not arg_val.get('complex'):
            return {arg_name.lower(): arg_val.get('val')}
        if arg_name.find('initializer') != -1:
            arg_fn = self._get_initializer(arg_val.get('name'))
            arg_fn_built = arg_fn(*arg_val.get('val'))
            return {arg_name.lower(): arg_fn_built}
        if arg_name.find('regularizer') != -1:
            arg_fn = self._get_regularizer(arg_val.get('name'))
            arg_fn_built = arg_fn(*arg_val.get('val'))
            return {arg_name.lower(): arg_fn_built}
        if arg_val.get('complex'):
            return {arg_name.lower(): {'layer': arg_val.get('name'), 'val': arg_val.get('val')}}
        return {arg_name, None}