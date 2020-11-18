from abc import ABC


class BaseMapper(ABC):
    def _config_mapper(self):
        self._default_arg = None

    def map(self, layer: dict):
        raise NotImplementedError("NOT IMPLEMENTED")