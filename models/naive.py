from sklearn.decomposition import PCA
import numpy as np

class NaiveCompression:
    def __init__(self):
        self.name = "Naive"
        self.input_shape = None
        self.avg = 0
        self.channels = 0

    def predict(self, X):
        return self.avg

    def training(self, X_train, Y_train = None, X_test = None, Y_test = None, p = None):
        self.avg = np.mean(X_train, axis=0)
        self.avg = self.avg.reshape(self.input_shape)