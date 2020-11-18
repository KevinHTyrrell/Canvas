from translators.arg_translator import ArgTranslator

layer_config = \
{
        'type': 'dense',
        'units': 200,
        'activation': ('leakyrelu', 0.2),
        'kernel_regularization': ('l1_l2', 1e-4, 1e-3),
        'dropout': 0.25,
        'flatten_input': True,
        'flatten_output': True
}

translator = ArgTranslator()
tmp = translator.feed_arg_dict(layer_config)
for k, v in tmp.items():
    print(k, v)