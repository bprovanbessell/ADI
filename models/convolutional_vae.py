import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras.layers import Input, Dense, Flatten, Reshape, Conv1D, Conv1DTranspose, BatchNormalization, Cropping1D
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam
from keras.callbacks import EarlyStopping
from keras import backend as K
from util import custom_keras

class ConvolutionalVAE:
    def __init__(self, model_path='../saved_models/'):
        self.name = "ConvVAE"
        self.model = None
        self.encoder = None
        self.decoder = None
        self.train_model = None
        self.train_encoder = None
        self.train_decoder = None
        self.best_val_loss = np.Inf
        self.input_shape = None
        self.latent_dim = 2
        self.count_save = 0
        self.model_path = model_path
        self.parameter_list = {'first_conv_layer_dim': [64, 128, 256, 512],
                           'first_window_size': [1, 3, 5, 7, 9],
                           'first_stride': [1, 3],
                           'conv_hidden_layers_5': [1, 2, 3, 4],
                           'conv_layer_dim_5': [64, 128, 256],
                           'window_size_5': [1, 3, 5, 7, 9],
                           'conv_additional_layer_5': [0, 1, 2, 3],
                           'conv_hidden_layers_2': [1, 2, 3],
                           'conv_layer_dim_2': [64, 128, 256],
                           'window_size_2': [1, 3, 5, 7, 9],
                           'conv_additional_layer_2': [0, 1, 2, 3],
                           'dense_hidden_layers': [1, 2, 3, 4, 5],
                           'dense_layer_dim': [32, 64, 128, 256],
                           'latent_dim': [4, 8, 16, 32],
                           'batch_size': [20],
                           'epochs': [1000],
                           'patience': [30],
                           'optimizer': ['adam', 'nadam', 'rmsprop'],
                           'conv_activation': ['relu', 'elu', 'selu'],
                           'dense_activation': ['relu', 'elu', 'selu'],
                           'lr': [1E-2, 1E-3, 1E-4],
                           'decay': [1E-2, 1E-3, 1E-4],
                           'conv_kernel_init': ['he_normal', 'glorot_uniform'],
                           'dense_kernel_init': ['he_normal', 'glorot_uniform']
                           }

    def predict(self, X):
        if self.model == None:
            print("ERROR: the model needs to be trained before predict")
            return
        return self.model.predict(X)

    def encode(self, X):
        if self.encoder == None:
            print("ERROR: the encoder needs to be trained before predict")
            return
        z, _, _ = self.encoder.predict(X)
        return z

    def decode(self, X):
        if self.decoder == None:
            print("ERROR: the encoder needs to be trained before predict")
            return
        return self.decoder.predict(X)

    def save_models(self):
        if self.model == None:
            print("ERROR: the model must be available before saving it")
            return
        self.model.save(self.model_path + self.name + str(self.count_save).zfill(4) + '_model.tf', save_format="tf")
        self.encoder.save(self.model_path + self.name + str(self.count_save).zfill(4)  + '_encoder.tf', save_format="tf")
        self.decoder.save(self.model_path + self.name + str(self.count_save).zfill(4)  + '_decoder.tf', save_format="tf")
        self.count_save += 1

    def load_models(self, name):
        self.model = load_model(self.model_path + name + '_model.tf', custom_objects={'Sampling': custom_keras.Sampling})
        self.encoder = load_model(self.model_path + name + '_encoder.tf', custom_objects={'Sampling': custom_keras.Sampling})
        self.decoder = load_model(self.model_path + name + '_decoder.tf')


    def training(self, X_train, Y_train, X_test, Y_test, p):
        """ Encoder and Decoder creation"""
        # Hyperparametrised VAE
        # Factorization of 15 000 = 2^3 x 3 x 5^4
        # Parameters for the first convolutional layer
        stride_5 = 5  # Fixed
        self.latent_dim = p['latent_dim']
        self.input_shape = X_train.shape[1:]
        # Parameters hidden convolutional layers of stride 2
        stride_2 = 2  # Fixed
        input_tensor = Input(shape=self.input_shape)
        # First con layer
        x = Conv1D(p['first_conv_layer_dim'], p['first_window_size'], activation=p['conv_activation'],
                   padding='same',
                   strides=p['first_stride'],
                   kernel_initializer=p['conv_kernel_init'])(input_tensor)
        x = BatchNormalization()(x)
        # Block of conv layers with stride 5
        for _ in range(p['conv_hidden_layers_5']):
            # Padding layers without stride
            for _ in range(p['conv_additional_layer_5']):
                x = Conv1D(p['conv_layer_dim_5'], p['window_size_5'], activation=p['conv_activation'],
                           padding='same',
                           kernel_initializer=p['conv_kernel_init'])(x)
                x = BatchNormalization()(x)
            x = Conv1D(p['conv_layer_dim_5'], p['window_size_5'], activation=p['conv_activation'],
                       padding='same',
                       strides=stride_5,
                       kernel_initializer=p['conv_kernel_init'])(x)
            x = BatchNormalization()(x)
        # Block of conv layers with stride 2
        for _ in range(p['conv_hidden_layers_2']):
            # Padding layers without stride
            for _ in range(p['conv_additional_layer_2']):
                x = Conv1D(p['conv_layer_dim_2'], p['window_size_2'], activation=p['conv_activation'],
                           padding='same',
                           kernel_initializer=p['conv_kernel_init'])(x)
                x = BatchNormalization()(x)
            x = Conv1D(p['conv_layer_dim_2'], p['window_size_2'], activation=p['conv_activation'],
                       padding='same',
                       strides=stride_2,
                       kernel_initializer=p['conv_kernel_init'])(x)
            x = BatchNormalization()(x)
        # Flattening of the tensor
        shape_before_flattening = K.int_shape(x)
        x = Flatten()(x)
        # Dense layers
        for _ in range(p['dense_hidden_layers']):
            x = Dense(p['dense_layer_dim'], activation=p['dense_activation'],
                      kernel_initializer=p['dense_kernel_init'])(x)
            x = BatchNormalization()(x)
        # Latent space layers
        z_mean = Dense(self.latent_dim, kernel_initializer=p['dense_kernel_init'])(x)
        z_log_var = Dense(self.latent_dim, kernel_initializer=p['dense_kernel_init'])(x)


        z = custom_keras.Sampling()([z_mean, z_log_var])
        # Encoder creation
        self.train_encoder = keras.Model(inputs=[input_tensor],
                              outputs=[z_mean, z_log_var, z], name="encoder")
        # Decoder
        latent_inputs = Input(K.int_shape(z)[1:])
        # Dense layer of the latent space
        x = Dense(p['dense_layer_dim'], activation=p['dense_activation'],
                  kernel_initializer=p['dense_kernel_init'])(latent_inputs)
        x = BatchNormalization()(x)
        # Dense layers
        for _ in range(p['dense_hidden_layers'] - 1):
            x = Dense(p['dense_layer_dim'], activation=p['dense_activation'],
                      kernel_initializer=p['dense_kernel_init'])(x)
            x = BatchNormalization()(x)
        x = Dense(np.prod(shape_before_flattening[1:]),
                  activation='relu',
                  kernel_initializer=p['dense_kernel_init'])(x)
        x = BatchNormalization()(x)
        # Reverse the flattening
        x = Reshape(shape_before_flattening[1:])(x)
        # Block of conv layers with stride 2
        for _ in range(p['conv_hidden_layers_2'] - 1):
            x = Conv1DTranspose(p['conv_layer_dim_2'], p['window_size_2'],
                                padding='same',
                                activation=p['conv_activation'],
                                strides=stride_2,
                                kernel_initializer=p['conv_kernel_init'])(x)
            x = BatchNormalization()(x)
            for _ in range(p['conv_additional_layer_2']):
                x = Conv1DTranspose(p['conv_layer_dim_2'], p['window_size_2'],
                                    padding='same',
                                    activation=p['conv_activation'],
                                    kernel_initializer=p['conv_kernel_init'])(x)
                x = BatchNormalization()(x)
        x = Conv1DTranspose(p['conv_layer_dim_2'], p['window_size_2'],
                            padding='same',
                            activation=p['conv_activation'],
                            strides=stride_2,
                            kernel_initializer=p['conv_kernel_init'])(x)
        x = BatchNormalization()(x)
        for _ in range(p['conv_additional_layer_2']):
            x = Conv1DTranspose(p['conv_layer_dim_2'], p['window_size_2'],
                                padding='same',
                                activation=p['conv_activation'],
                                kernel_initializer=p['conv_kernel_init'])(x)
            x = BatchNormalization()(x)
        # Block of conv layers with stride 5
        for _ in range(p['conv_hidden_layers_5'] - 1):
            x = Conv1DTranspose(p['conv_layer_dim_5'], p['window_size_5'],
                                padding='same',
                                activation=p['conv_activation'],
                                strides=stride_5,
                                kernel_initializer=p['conv_kernel_init'])(x)
            x = BatchNormalization()(x)
            for _ in range(p['conv_additional_layer_5']):
                x = Conv1DTranspose(p['conv_layer_dim_5'], p['window_size_5'],
                                    padding='same',
                                    activation=p['conv_activation'],
                                    kernel_initializer=p['conv_kernel_init'])(x)
                x = BatchNormalization()(x)
        x = Conv1DTranspose(p['first_conv_layer_dim'], p['window_size_5'],
                            padding='same',
                            activation=p['conv_activation'],
                            strides=stride_5,
                            kernel_initializer=p['conv_kernel_init'])(x)
        x = BatchNormalization()(x)
        for _ in range(p['conv_additional_layer_5']):
            x = Conv1DTranspose(p['first_conv_layer_dim'], p['window_size_5'],
                                padding='same',
                                activation=p['conv_activation'],
                                kernel_initializer=p['conv_kernel_init'])(x)
            x = BatchNormalization()(x)
        output_tensor = Conv1DTranspose(self.input_shape[1], p['first_window_size'],
                                        padding='same',
                                        strides=p['first_stride'],
                                        kernel_initializer=p['conv_kernel_init'])(x)
        shape_err = K.int_shape(output_tensor)[1] - self.input_shape[0]
        if shape_err > 0:
            print("A crop is needed of ", shape_err)
            output_tensor = Cropping1D(cropping=(0, shape_err))(output_tensor)
        self.train_decoder = keras.Model(latent_inputs, output_tensor, name="decoder")
        """ Model creation """
        _, _, z = self.train_encoder(input_tensor)
        reconstructions = self.train_decoder(z)
        self.train_model = keras.Model(inputs=[input_tensor], outputs=[reconstructions])
        # self.encoder.summary()
        # self.decoder.summary()
        # self.model.summary()
        # Loss definition
        latent_loss = -0.5 * K.sum(
            1 + z_log_var - K.exp(z_log_var) - K.square(z_mean), axis=-1
        )
        # we do it to ensure the appropriate scale between it and the reconstruction loss
        scale_factor = self.input_shape[0]  # reduce it to give more important to the compactness of the latent space
        self.train_model.add_loss(K.mean(latent_loss) / scale_factor)
        # Model optimizer
        opt = None
        if p['optimizer'] == 'adam':
            opt = Adam(lr=p['lr'], decay=p['decay'])
        elif p['optimizer'] == 'rmsprop':
            opt = RMSprop(learning_rate=p['lr'])
        elif p['optimizer'] == 'nadam':
            opt = Nadam(learning_rate=p['lr'])
        # Model compilation
        self.train_model.compile(loss='mse', optimizer=opt,
                           metrics=["mean_absolute_error", "mean_absolute_percentage_error"])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=p['patience'])
        # mcp_save = ModelCheckpoint(path_to_data +'mdl_vae_flux_private.h5', save_best_only=True, monitor='val_loss', mode='min')
        vae_save = custom_keras.VAESaveCheckpoint(self)
        # Training
        result = self.train_model.fit(X_train, X_train, epochs=p['epochs'], batch_size=p['batch_size'], validation_split=0.2,
                                callbacks=[es, vae_save], verbose=2)
        validation_loss = np.amin(result.history['val_loss'])
        print('Best validation loss of epoch:', validation_loss)
        return result, self.train_model
