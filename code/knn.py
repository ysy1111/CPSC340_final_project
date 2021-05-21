"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k, method="euclidean"):
        self.k = k
        self.method = method

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y 

    

    def predict(self, Xtest):
        t, D = Xtest.shape
        N, D = self.X.shape
        if self.method == "euclidean":
            DMatrix=utils.euclidean_dist_squared(self.X,Xtest)
        elif self.method == "cos_sim":
            DMatrix= self.cosine_distance(self.X,Xtest)
        # elif self.method == "RBF_kernel":
        #     DMatrix=utils.RBF_kernel_dist(self.X,Xtest)
        y_pred=np.zeros(t)
        for n in range(t):
            knn_index=np.argsort(DMatrix[:,n])[0:self.k]
            knn_label=np.zeros(self.k)
            for i in range(self.k):
                knn_label[i]=self.y[knn_index[i]]
            y_pred[n]=utils.mode(knn_label)

        return y_pred

    def cosine_distance(self,X1,X2):
        cos_sim = X1@X2.T/(np.sqrt(np.sum(X1**2,axis=1))[:,None]@np.sqrt(np.sum(X2**2,axis=1))[None])
        where_are_NaNs = np.isnan(cos_sim)
        cos_sim[where_are_NaNs] = 0
        return 1-cos_sim