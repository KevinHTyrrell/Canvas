import numpy as np
import random
from tensorflow import keras
from keras.datasets import mnist
from workers.architect import Architect

latent_dims = 100
start_img_dims = (7, 7)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=len(x_train.shape))
x_test = np.expand_dims(x_test, axis=len(x_test.shape))
gen_config_list = [
        {
                  'type'                : 'dense'
                , 'units'               : int(np.prod(start_img_dims) * latent_dims)
                , 'activation'          : ('leakyrelu', 0.2)
                , 'kernel_initializer'  : ('RandomUniform', -0.5, 0.5, 123456)
                , 'kernel_regularizer'  : ('l1l2', 1e-4, 1e-4)
                , 'dropout'             : 0.25
                , 'reshape'             : (7, 7, latent_dims)
        },
        {
                'type': 'conv2dtranspose'
                , 'filters': 128
                , 'kernel_size': (4, 4)
                , 'strides': (1, 1)
                , 'activation': ('leakyrelu', 0.2)
                , 'kernel_initializer': ('RandomUniform', -0.5, 0.5, 123456)
                , 'kernel_regularizer': ('l1l2', 1e-4, 1e-4)
                , 'padding': 'same'
                , 'batch_norm': True
                , 'dilution': ('gaussiannoise', 0.01)
                , 'upsample_input': (2, 2)
        },
        {
                'type': 'conv2dtranspose'
                , 'filters': 128
                , 'kernel_size': (4, 4)
                , 'strides': (1, 1)
                , 'activation': ('leakyrelu', 0.2)
                , 'kernel_initializer': ('RandomUniform', -0.5, 0.5, 123456)
                , 'kernel_regularizer': ('l1l2', 1e-4, 1e-4)
                , 'padding': 'same'
                , 'batch_norm': True
                , 'dilution': ('gaussiannoise', 0.01)
                , 'upsample_input': (2, 2)
        },
        {
                'type': 'conv2d'
                , 'filters': 1
                , 'kernel_size': (4, 4)
                , 'strides': (1, 1)
                , 'activation': ('leakyrelu', 0.2)
                , 'kernel_initializer': ('RandomUniform', -0.5, 0.5, 123456)
                , 'kernel_regularizer': ('l1l2', 1e-4, 1e-4)
                , 'padding': 'same'
        },
    ]

disc_config_list = [
        {
                'type': 'conv2d'
                , 'filters': 32
                , 'kernel_size': (2, 2)
                , 'strides': (1, 1)
                , 'activation': ('leakyrelu', 0.2)
                , 'kernel_initializer': ('RandomUniform', -0.5, 0.5, 123456)
                , 'kernel_regularizer': ('l1l2', 1e-4, 1e-4)
                , 'padding': 'same'
                , 'max_pool': (2, 2)
        },
        {
                'type': 'conv2d'
                , 'filters': 16
                , 'kernel_size': (2, 2)
                , 'strides': (1, 1)
                , 'activation': ('leakyrelu', 0.2)
                , 'kernel_initializer': ('RandomUniform', -0.5, 0.5, 123456)
                , 'kernel_regularizer': ('l1l2', 1e-4, 1e-4)
                , 'padding': 'same'
                , 'max_pool': (2, 2)
        },
        {
                'type': 'dense'
                , 'units': 100
                , 'activation': ('leakyrelu', 0.2)
                , 'kernel_initializer': ('RandomUniform', -0.5, 0.5, 123456)
                , 'kernel_regularizer': ('l1l2', 1e-4, 1e-4)
                , 'dropout': 0.25
                , 'flatten_input': True
        },
        {
                'type': 'dense'
                , 'units': 1
                , 'activation': 'sigmoid'
                , 'kernel_initializer': ('RandomUniform', -0.5, 0.5, 123456)
                , 'kernel_regularizer': ('l1l2', 1e-4, 1e-4)
        }
    ]

architect = Architect()
gen_tensor_list = architect.build_model(input_shape=(100,), layer_config_list=gen_config_list)
gen_output_shape = list(gen_tensor_list[-1].shape)[1:]
disc_tensor_list = architect.build_model(input_shape=tuple(gen_output_shape), layer_config_list=disc_config_list)

my_gen = keras.models.Model(inputs=gen_tensor_list[0], outputs=gen_tensor_list[-1])

my_disc = keras.models.Model(inputs=disc_tensor_list[0], outputs=disc_tensor_list[-1])
my_disc.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

gan_input = keras.layers.Input(shape=latent_dims)
gan_gen_out = my_gen(gan_input)
gan_disc_out = my_disc(gan_gen_out)
full_gan = keras.Model(inputs=gan_input, outputs=gan_disc_out)
full_gan.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

n_episodes = 500
n_epochs_disc = 5
n_epochs_gen = 3
half_sample = 2500  # avoid coercing a float to int
for n_episode in range(n_episodes):
        print('Episode: {n_episode}'.format(n_episode=n_episode))
        print('discriminator')
        x_disc_seed = np.random.random(latent_dims * half_sample)
        x_disc_seed_reshaped = np.reshape(x_disc_seed, newshape=(half_sample, latent_dims))
        x_disc_fake = my_gen.predict(x_disc_seed_reshaped)
        y_disc_fake = np.zeros(half_sample)
        my_disc.trainable = True

        for disc_epoch in range(n_epochs_disc):
                x_disc_real = np.asarray(random.sample(list(x_train), half_sample))
                y_disc_real = np.ones(half_sample)
                x_disc_train = np.vstack([x_disc_real, x_disc_fake])
                y_disc_train = np.concatenate([y_disc_real, y_disc_fake])
                my_disc.fit(x=x_disc_train, y=y_disc_train, shuffle=True)

        # train generator #
        print('generator')
        my_disc.trainable = False
        for gen_epoch in range(n_epochs_gen):
                x_gen_seed = np.random.random(latent_dims * half_sample * 2)
                x_gen_seed_reshaped = np.reshape(x_gen_seed, newshape=(2 * half_sample, latent_dims))
                x_gen_train = x_gen_seed_reshaped
                y_gen_train = np.ones(2 * half_sample)
                full_gan.fit(x=x_gen_train, y=y_gen_train, shuffle=True)

        test_img_seed = np.expand_dims(np.random.random(latent_dims), axis=0)
        test_img = my_gen.predict(test_img_seed).squeeze()
        test_img_int = test_img.astype(int)
