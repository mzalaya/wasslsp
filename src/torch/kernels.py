# Author: Mokhtar Z. Alaya <alayaelm@utc.fr>
# License:

import numpy as np
import torch
import functorch

from sklearn.base import BaseEstimator

from src.torch.utils import(
uniform, rectangle, triangle,
epanechnikov, biweight, tricube,
gaussian, silverman
)


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
    vectorize_kernel = torch.func.vmap(kernel)
    kernel_vec_val = vectorize_kernel(x_Xt_scaled)

    return torch.prod(kernel_vec_val)

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
            device="cpu",
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
        self.device = device
        self.VALID_KERNELS_DIC = VALID_KERNELS_DIC

        if self.space_kernel in VALID_KERNELS_LIST and self.time_kernel in VALID_KERNELS_LIST:
            self.skernel_name = self.VALID_KERNELS_DIC[self.space_kernel]
            self.tkernel_name = self.VALID_KERNELS_DIC[self.time_kernel]
        else:
            raise ValueError("Kernel type not supported")

    def fit(self, x, t):

        time_vals = [time_kernel(self.tkernel_name, a/self.T, t/self.T, self.bandwidth) for a in range(self.T)]

        space_vals = [space_kernel(self.skernel_name, x[t], x[a], self.bandwidth) for a in range(self.T)]

        ts_vals = torch.tensor(time_vals) * torch.tensor(space_vals)

        weights_t = ts_vals / ts_vals.sum()
        return weights_t.to(self.device)