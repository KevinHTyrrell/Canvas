from experts.arg_expert import ArgExpert
from translators.base_translator import BaseTranslator


class ArgTranslator(BaseTranslator):
    def _config(self):
        self._expert = ArgExpert()

    def feed_arg_dict(self, args: dict):
        # check that each arg is a valid input, if so then parse accordingly and add to arg_dict #
        for arg_name, arg_val in args.items():
            self._expert.check_arg(arg_name, arg_val)
            self._translations.update({arg_name: getattr(self, '_parse_' + type(arg_val).__name__)(arg_val)})
        return self._translations

    def _parse_bool(self, arg_val):
        return {'val': arg_val, 'complex': False}

    def _parse_float(self, arg_val):
        return {'val': arg_val, 'complex': False}

    def _parse_int(self, arg_val):
        return {'val': arg_val, 'complex': False}

    def _parse_str(self, arg_val):
        return {'val': arg_val, 'complex': False}

    def _parse_tuple(self, arg_val):
        return_dict = dict()
        complex_val = False
        str_args = [arg for arg in arg_val if type(arg) == str]
        num_args = [arg for arg in arg_val if type(arg) == int or type(arg) == float]
        if len(str_args) > 0: # string in tuple
            return_dict.update({'name': str_args[0]})
        if len(num_args) > 0: # nums in tuple
            complex_val = True
            return_dict.update({'val': num_args})
        return_dict.update({'complex': complex_val})
        return return_dict
