from builders.block.base_block import BaseBlock, keras


class TransferBlock(BaseBlock):
    def build(self, input_layer: keras.layers.Layer):
        # returns last layer to feed into next block #
        pass