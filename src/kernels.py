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

def space_kernel(kernel, x, Xt, bandwidth):
    """
    Space kernel
    :param kernel: function
    :param x: single point in R^d
    :param Xt: single data point in R^d at time t
    :param bandwidth: float
    :return:
    """
    x_Xt_scaled = (x - Xt) / bandwidth
    vectorize_kernel = np.vectorize(kernel)
    kernel_vec_val = vectorize_kernel(x_Xt_scaled)
    return np.prod(kernel_vec_val)

def time_kernel(kernel, aT, tT, bandwidth):
    """
    Time kernel
    :param kernel:
    :param aT:
    :param tT:
    :param bandwidth:
    :return:
    """
    atT_scaled = (tT - aT) / bandwidth
    return kernel(atT_scaled)

class Kernel(BaseEstimator):
    def __init__(
            self,
            *,
            T=100,
            d=2,
            bandwidth=1.0,
            space_kernel="gaussian",
            time_kernel="gaussian",
            metric="euclidean",
            VALID_KERNELS_DIC={
                "uniform": uniform,
                "rectangle": rectangle,
                "triangle": triangle,
                "epanechnikov": epanechnikov,
                "biweight": biweight,
                "tricube": tricube,
                "gaussian": gaussian,
                "silverman": silverman,
            }
    ):
        self.T = T
        self.d = d
        self.bandwidth = bandwidth
        self.space_kernel = space_kernel
        self.time_kernel = time_kernel
        self.metric = metric

        self.VALID_KERNELS_DIC = VALID_KERNELS_DIC

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

        if self.space_kernel in VALID_KERNELS_LIST and self.time_kernel in VALID_KERNELS_LIST:
            for a in range(self.T):
                Xa = X[a]
                aT = a / self.T
                skernel_name = self.VALID_KERNELS_DIC[self.space_kernel]
                tkernel_name = self.VALID_KERNELS_DIC[self.time_kernel]
                space_val = space_kernel(skernel_name , x, Xa, self.bandwidth)
                time_val = time_kernel(tkernel_name, aT, tT, self.bandwidth)
                numerator.append(space_val * time_val)
                denominator.append(space_val * time_val)

        # if self.space_kernel == "uniform" and self.time_kernel == "uniform":
        #     for a in range(self.T):
        #         Xa = X[a]
        #         aT = a / self.T
        #         space_unif_val = space_kernel(uniform, x, Xa, self.bandwidth)
        #         time_unif_val = time_kernel(uniform, aT, tT, self.bandwidth)
        #         numerator.append(space_unif_val * time_unif_val)
        #         denominator.append(space_unif_val * time_unif_val)
        #
        # elif self.space_kernel == "epanechnikov" and self.time_kernel == "epanechnikov":
        #     for a in range(self.T):
        #         Xa = X[a]
        #         aT = a / self.T
        #         space_epan_val = space_kernel(epanechnikov, x, Xa, self.bandwidth)
        #         time_epan_val = time_kernel(epanechnikov, aT, tT, self.bandwidth)
        #         numerator.append(space_epan_val * time_epan_val)
        #         denominator.append(space_epan_val * time_epan_val)
        #
        # elif self.space_kernel == "gaussian" and self.time_kernel == "gaussian":
        #     for a in range(self.T):
        #         Xa = X[a]
        #         aT = a / self.T
        #         space_gauss_val = space_kernel(gaussian, x, Xa, self.bandwidth)
        #         time_gauss_val = time_kernel(gaussian, aT, tT, self.bandwidth)
        #         numerator.append(space_gauss_val * time_gauss_val)
        #         denominator.append(space_gauss_val * time_gauss_val)
        #
        # elif self.space_kernel == "silverman" and self.time_kernel == "silverman":
        #     for a in range(self.T):
        #         Xa = X[a]
        #         aT = a / self.T
        #         space_silv_val = space_kernel(silverman, x, Xa, self.bandwidth)
        #         time_silv_val = time_kernel(silverman, aT, tT, self.bandwidth)
        #         numerator.append(space_silv_val * time_silv_val)
        #         denominator.append(space_silv_val * time_silv_val)
        #
        # elif self.space_kernel == "tricube" and self.time_kernel == "tricube":
        #     for a in range(self.T):
        #         Xa = X[a]
        #         aT = a / self.T
        #         space_tricube_val = space_kernel(tricube, x, Xa, self.bandwidth)
        #         time_tricube_val = time_kernel(tricube, aT, tT, self.bandwidth)
        #         numerator.append(space_tricube_val * time_tricube_val)
        #         denominator.append(space_tricube_val * time_tricube_val)
        #
        # elif self.space_kernel == "biweight" and self.time_kernel == "biweight":
        #     for a in range(self.T):
        #         Xa = X[a]
        #         aT = a / self.T
        #         space_biweight_val = space_kernel(biweight, x, Xa, self.bandwidth)
        #         time_biweight_val = time_kernel(biweight, aT, tT, self.bandwidth)
        #         numerator.append(space_biweight_val * time_biweight_val)
        #         denominator.append(space_biweight_val * time_biweight_val)

        else:
            raise ValueError("Kernel type not supported")
            
        weights_t = np.array(numerator) / np.array(denominator).sum()
        #self.weights_X[t] = weights_t
        return weights_t #self.weights_X