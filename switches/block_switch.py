# WILL TAKE IN DICTIONARY OF PARSED ARGUMENTS AND PASS OUT ACCORDINGLY #
from switches.base_switch import BaseOperator


class BlockOperator(BaseOperator):
    def _config(self):
        self._arg_table = {
            'transform' : list(),
            'train'     : list(),
            'transfer'  : list()
        }

    def set_transform_args(self, args: list) -> None:
        self._arg_table.update({'transform': args})
    def set_train_args(self, args: list) -> None:
        self._arg_table.update({'transform': args})
    def set_transfer_args(self, args: list) -> None:
        self._arg_table.update({'transform': args})