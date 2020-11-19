from builders.block.base_block import BaseBlock, keras


class TrainBlock(BaseBlock):
    def build(self, input_layer: keras.layers.Layer):
        # take type out of hyperparameters and call that layer #
        # returns last layer to feed into next block #
        pass