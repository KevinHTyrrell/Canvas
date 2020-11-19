from builders.block.base_block import BaseBlock, keras


class TransformBlock(BaseBlock):
    def build(self, input_layer: keras.layers.Layer):
        hyperparameters = self._args.get('hyperparameters')
        print(hyperparameters)
        pass