import numpy as np

import keras
from keras import Sequential, optimizers, regularizers
from keras.layers import Dense

from rdkit.Chem import Descriptors, MolFromSmiles
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class pIC50_predictor(BaseEstimator):
    def __init__(self, l1=0.0, l2=0.0, hn1=100, hn2=100, learning_rate=0.0001, batch_size=10,  epochs=100):
       
        self.l1 = l1 
        self.l2 = l2
        self.hn1 = hn1 
        self.hn2 = hn2
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X, y):

        X, y = check_X_y(X, y, accept_sparse=False)

        n_feat = X.shape[1]

        model = Sequential()
        model.add(Dense(self.hn1, input_dim=n_feat, activation='tanh', kernel_regularizer=regularizers.l1_l2(self.l1, self.l2)))
        model.add(Dense(self.hn2, activation='tanh', kernel_regularizer=regularizers.l1_l2(self.l1, self.l2)))
        model.add(Dense(1, activation='linear'))
        optimiser = optimizers.Adam(lr=self.learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimiser)
        
        model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs)
        self._model = model
        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, 'is_fitted_')

        y_pred = self._model.predict(X)
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            return y_pred.ravel()
        else:
            return y_pred

    def score(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False)
        X = check_array(X, accept_sparse=False)

        y_pred = self._model.predict(X)

        score = r2_score(y,y_pred)

        return score

    def save(self, filename="pic50_model.hdf5"):
        if self.is_fitted_:
            self._model.save(filename, overwrite=True)
