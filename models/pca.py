from sklearn.decomposition import PCA
import numpy as np

class PCAModel:
    def __init__(self):
        self.name = "PCAModel"
        self.input_shape = None
        self.latent_dim = 8
        self.pca = None
        self.channels = 0

    def predict(self, X):
        if self.pca is None:
            print("ERROR: the model needs to be trained before predict")
            return
        return self.decode(self.encode(X))

    def encode(self, X):
        if self.pca is None:
            print("ERROR: the encoder needs to be trained before predict")
            return
        out = np.zeros((X.shape[0], self.latent_dim, self.channels))
        for i in range(self.channels):
            out[:,:,i] = self.pca[i].transform(X[:,:,i])
        return out

    def decode(self, X):
        if self.pca is None:
            print("ERROR: the encoder needs to be trained before predict")
            return
        out = np.zeros((X.shape[0], 15000, self.channels))
        for i in range(self.channels):
            out[:,:,i] = self.pca[i].inverse_transform(X[:,:,i])
        return out

    # def save_models(self, path, name):
    #     if self.model == None:
    #         print("ERROR: the model must be available before saving it")
    #         return
    #     self.model.save(path + name + '_model.tf', save_format="tf")
    #     self.encoder.save(path + name + '_encoder.tf', save_format="tf")
    #     self.decoder.save(path + name + '_decoder.tf', save_format="tf")
    #
    # def load_models(self, path, name):
    #     self.model = load_model(path + name + '_model.tf', custom_objects={'Sampling': sampling.Sampling})
    #     self.encoder = load_model(path + name + '_encoder.tf', custom_objects={'Sampling': sampling.Sampling})
    #     self.decoder = load_model(path + name + '_decoder.tf')

    def training(self, X_train, Y_train = None, X_test = None, Y_test = None, p = None):
        self.pca = {}
        self.channels = X_train.shape[2]
        for i in range(self.channels):
            self.pca[i] = PCA(n_components=self.latent_dim)
            self.pca[i].fit(X_train[:,:,i])

