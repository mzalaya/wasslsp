# Author: Mokhtar Z. Alaya <alayaelm@utc.fr>
# License:

import torch

import numpy as np
import torch
import functorch

from sklearn.base import BaseEstimator


from src.torch.utils import (
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
    :param Xt: data points in R^d at time t (batched)
    :param bandwidth: float
    :return:
    """
    x_Xt_scaled = (x - Xt) / bandwidth
    kernel_vec_val = torch.vmap(kernel)(x_Xt_scaled)
    return torch.prod(kernel_vec_val, dim=-1)

def time_kernel(kernel, aT, tT, bandwidth):
    """
    Time kernel
    :param kernel: function
    :param aT: scaled time values (batched)
    :param tT: target time value
    :param bandwidth: float
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
        # Generate time indices for all points
        a = torch.arange(self.T, dtype=torch.float32, device=self.device)

        # Calculate time kernel values in a vectorized manner
        time_vals = time_kernel(self.tkernel_name, a / self.T, t / self.T, self.bandwidth)

        # Calculate space kernel values in a vectorized manner
        space_vals = space_kernel(self.skernel_name, x[t], x, self.bandwidth)

        # Calculate the combined kernel values
        ts_vals = time_vals * space_vals

        # Normalize the weights
        weights_t = ts_vals / ts_vals.sum()
        return weights_t.to(self.device)
