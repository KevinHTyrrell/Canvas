from tensorflow import keras
from translators.arg_translator import ArgTranslator
from operators.block_operator import BlockOperator
from builders.block.transform_block import TransformBlock
from builders.block.train_block import TrainBlock
from builders.block.transfer_block import TransferBlock


class Architect:
    def __init__(self):
        self._arg_translator        = ArgTranslator()
        self._block_operator        = BlockOperator()   # used to communicate with each block

    def build_model(self, input_shape: tuple, layer_config_list: list):
        tensor_list = list()
        current_layer = keras.layers.Input(input_shape)
        block_list = [TransformBlock(), TrainBlock(), TransferBlock()]
        for layer_config in layer_config_list:
            translated_hyperparameters = self._arg_translator.feed_arg_dict(layer_config)
            self._block_operator.set_blocks(*block_list)
            self._block_operator.assign_hyperparameters(translated_hyperparameters)
            tensor_list += self._block_operator.build_blocks(current_layer)
            return block_list, tensor_list