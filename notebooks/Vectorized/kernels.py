# Author: Mokhtar Z. Alaya <alayaelm@utc.fr>
# License:
import sys
sys.path.append("/Users/mzalaya/Library/CloudStorage/Dropbox/research/git/wasslsp/src/")
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
VALID_KERNELS_LIST = [
    "uniform",
    "rectangle",
    "triangle",
    "epanechnikov",
    "biweight",
    "tricube",
    "gaussian",
    "silverman",
]

def space_kernel(kernel, x, X, bandwidth):
    """
    Vectorized space kernel: computes kernel values for a batch of points
    :param kernel: Kernel function capable of vectorized operations
    :param x: single point in R^d
    :param X: Array of points in R^d
    :param bandwidth: float
    :return: Array of kernel values
    """
    x_X_scaled = (x - X) / bandwidth
    return np.prod(kernel(x_X_scaled), axis=1)

def time_kernel(kernel, aT, tT, bandwidth):
    """
    Vectorized time kernel: computes kernel values for a batch of times
    :param kernel: Kernel function capable of vectorized operations
    :param aT: Array of times
    :param tT: Single time point
    :param bandwidth: float
    :return: Array of kernel values
    """
    atT_scaled = (tT - aT) / bandwidth
    return kernel(atT_scaled)

class Kernel(BaseEstimator):
    def __init__(self, *, T=100, d=2, bandwidth=1.0, space_kernel="gaussian", time_kernel="gaussian", metric="euclidean"):
        self.T = T
        self.d = d
        self.bandwidth = bandwidth
        self.space_kernel = space_kernel
        self.time_kernel = time_kernel
        self.metric = metric
        self.VALID_KERNELS_DIC = {
            "uniform": uniform, "rectangle": rectangle, "triangle": triangle,
            "epanechnikov": epanechnikov, "biweight": biweight, "tricube": tricube,
            "gaussian": gaussian, "silverman": silverman,
        }

    def fit(self, X, t):
        """
        Vectorized Fit method for Kernel Density Estimation
        :param X: array-like of shape (n_samples, n_features)
        :param t: Time index for which the weight is to be calculated
        :return: Weights for the given time t
        """
        tT = t / self.T
        x = X[t]
        aT = np.arange(self.T) / self.T
        
        space_val = space_kernel(self.VALID_KERNELS_DIC[self.space_kernel], x, X, self.bandwidth)
        time_val = time_kernel(self.VALID_KERNELS_DIC[self.time_kernel], aT, tT, self.bandwidth)
        
        weights = space_val * time_val
        
        return weights / weights.sum()
    