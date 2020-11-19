from mappers.base_mapper import BaseMapper


class ArgMapper(BaseMapper):
    def _config(self):
        self._load_initializers()
        self._load_regularizers()

    def _load_initializers(self):
        init_dict = dict()
        for init_name, init_fn in keras.initializers.__dict__.items():
            if isinstance(init_fn, type):
                init_dict.update({init_name: init_fn})
        self._args['initializers'] = init_dict

    def _load_regularizers(self):
        reg_dict = dict()
        for init_name, init_fn in keras.regularizers.__dict__.items():
            if isinstance(init_fn, type):
                reg_dict.update({init_name: init_fn})
        self._args['regularizers'] = reg_dict

    def get_initializer(self, init_name):
        return self._args['initializers'].get(init_name.lower())

    def get_regularizer(self, reg_name):
        return self._args['regularizers'].get(reg_name.lower())