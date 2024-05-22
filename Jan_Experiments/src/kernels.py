import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

from utils import(
uniform, rectangle, triangle,
epanechnikov, biweight, tricube,
gaussian, silverman
)
"""
Kernel module
Ref: https://github.com/scikit-learn/scikit-learn/blob/872124551/sklearn/neighbors/_kde.py#L35
"""
VALID_KERNELS = [
    "uniform",
    "rectangle",
    "triangle",
    "epanechnikov",
    "biweight",
    "tricube",
    "gaussian",
    "silverman",
]

def space_kernel(kernel, x, Xt, bandwith):
    """
    :param kernel: function
    :param x: single point in R^d
    :param Xt: single data point in R^d at time t
    :param bandwith: float
    :return:
    """
    x_Xt_scaled = (x - Xt) / bandwith
    vectorize_kernel = np.vectorize(kernel)
    kernel_vec_val = vectorize_kernel(x_Xt_scaled)
    # kernel_vec_val = kernel(x_Xt_scaled)
    return np.prod(kernel_vec_val)

def time_kernel(kernel, aT, tT, bandwith):
    atT_scaled = (tT - aT) / bandwith
    return kernel(atT_scaled)

class Kernel(BaseEstimator):
    def __init__(
            self,
            *,
            T=100,
            d=3,
            bandwith=1.0,
            space_kernel="gaussian",
            time_kernel="gaussian",
            metric="euclidean",
    ):
        self.T = T
        self.d = d
        self.bandwith = bandwith
        self.space_kernel = space_kernel
        self.time_kernel = time_kernel
        self.metric = metric


    def fit(self, X, t):
        """
        Fit the Kernel Density model on the data.
        :param X: array-like of shape (n_samples, n_features)
                List of n_features-dimensional data points.  Each row corresponds to a single data point.
        :param y:
        :return:
        """
        #self.weights_X = np.zeros(self.T)
        tT = t / self.T
        x = X[t]
        numerator = []
        denominator = []

        if self.space_kernel == "uniform" and self.time_kernel == "uniform":
            for a in range(self.T):
                Xa = X[a]
                aT = a / self.T
                space_unif_val = space_kernel(uniform, x, Xa, self.bandwith)
                time_unif_val = time_kernel(uniform, aT, tT, self.bandwith)
                numerator.append(space_unif_val * time_unif_val)
                denominator.append(space_unif_val * time_unif_val)

        elif self.space_kernel == "epanechnikov" and self.time_kernel == "epanechnikov":
            for a in range(self.T):
                Xa = X[a]
                aT = a / self.T
                space_epan_val = space_kernel(epanechnikov, x, Xa, self.bandwith)
                time_epan_val = time_kernel(epanechnikov, aT, tT, self.bandwith)
                numerator.append(space_epan_val * time_epan_val)
                denominator.append(space_epan_val * time_epan_val)

        elif self.space_kernel == "gaussian" and self.time_kernel == "gaussian":
            for a in range(self.T):
                Xa = X[a]
                aT = a / self.T
                space_gauss_val = space_kernel(gaussian, x, Xa, self.bandwith)
                time_gauss_val = time_kernel(gaussian, aT, tT, self.bandwith)
                numerator.append(space_gauss_val * time_gauss_val)
                denominator.append(space_gauss_val * time_gauss_val)

        elif self.space_kernel == "silverman" and self.time_kernel == "silverman":
            for a in range(self.T):
                Xa = X[a]
                aT = a / self.T
                space_silv_val = space_kernel(silverman, x, Xa, self.bandwith)
                time_silv_val = time_kernel(silverman, aT, tT, self.bandwith)
                numerator.append(space_silv_val * time_silv_val)
                denominator.append(space_silv_val * time_silv_val)
        else:
            raise ValueError("Kernel type not supported")
            
        weights_t = np.array(numerator) / np.array(denominator).sum()
        #self.weights_X[t] = weights_t
        return weights_t #self.weights_X