#!/usr/bin/env python
# coding: utf-8
# Author: Mokhtar Z. Alaya <alayaelm@utc.fr>

"""
Gaussian tvTAR(1)
Ref: S. Richter and R. Dahlhaus. Cross validation for locally stationary processes. Ann. Statist., 47(4):2145–2173, 2019
"""
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')
from numba import njit
from numba import cuda

import time
import numpy as np
import torch

from tensordict import TensorDict

import platform

import pickle

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

if platform.system() == 'Darwin':
    params = {'axes.labelsize': 12,
              'font.size': 12,
              'legend.fontsize': 12,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12, # 10
              'text.usetex': True,
              'figure.figsize': (10, 8)}
    plt.rcParams.update(params)
else:
    params = {'axes.labelsize': 12,
              'font.size': 12,
              'legend.fontsize': 12,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12, # 10
              'figure.figsize': (10, 8)}
    plt.rcParams.update(params)

import sys
sys.path.append("/Users/mzalaya/Documents/git/wasslsp/")

from src.torch.utils import *
from src.torch.utils import ECDFTorch
from src.torch.kernels import Kernel

from scipy.stats import wasserstein_distance


def running_test(test, device):
    if test is not None:
        times_t = np.array([1000, 1050, 1100, 1200, 1250, 1300, 1350, 1400], dtype=int)
        # times_T = np.array([2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000], dtype=int)
        times_T = np.array([2000, 6000, 10000, 14000, 18000, 22000, 26000, 30000], dtype=int)
        n_replications = 1000
    else:
        times_t =  np.array([1000, 1050, 1100, 1200, 1250, 1300, 1350, 1400], dtype=int)
        times_T =  np.array([2000, 3000, 4000, 5000], dtype=int)
        n_replications = 10

    print(
        f'times_t [{times_t}] | \n'
        f'times_T = {times_T} | \n'
        f'n_replications = {n_replications} | device = {device}'
    )

    return times_t, times_T, n_replications

# phi_star = lambda u: 1.5 - torch.cos(2 * torch.pi * torch.tensor([u]))
@njit
def phi_star(u):
    return 1.5 - np.cos(2 * np.pi * u)
# phi_star = lambda u: 1.5 - np.cos(2 * np.pi * u)


# @cuda.jit
@njit
def conditional_mean_function(u, x, y):
    # m_star = lambda u, x, y: psi_star(u) * torch.cos(phi_star(u)) * x - 0.8 * y
    m_star = lambda u, x, y: 1.8 * np.cos(phi_star(u)) * x - 0.8 * y

    return m_star(u, x, y)

# @cuda.jit(device=True)
@njit
def simulation_1_rep_process(d, T):
    t = 2
    epsilon = np.random.normal(0., 1., (T,))
    X = np.zeros((T,d))
    X_tvar_2_T = np.zeros(T)
    while t <= T - 1:
        m_star_val = conditional_mean_function(t/T, X_tvar_2_T[t - 1], X_tvar_2_T[t - 2])
        X_tvar_2_T[t] = m_star_val + 1.0 * epsilon[t]
        X[t] = np.array([X_tvar_2_T[t - 1], X_tvar_2_T[t - 2]])
        t += 1

    return X_tvar_2_T, X

def _simulation_1_rep_process(d, T, device):
    t = 2
    epsilon = torch.normal(mean=0., std=1., size=(T,)).to(device)
    X = torch.zeros((T, d)).to(device)
    X_tvar_2_T = torch.zeros(T).to(device)

    while t <= T - 1:
        m_star_val = conditional_mean_function(t / T, X_tvar_2_T[t - 1].to(device='cpu'),
                                               X_tvar_2_T[t - 2].to(device='cpu'))
        X_tvar_2_T[t] = m_star_val.to(device) + 1.0 * epsilon[t]
        X[t] = torch.tensor([X_tvar_2_T[t - 1], X_tvar_2_T[t - 2]])
        t += 1

    return X_tvar_2_T, X


def simulation_L_rep_process(d, times_t, times_T, n_replications, device):
    tic = datetime.now()
    print('-' * 100)
    print("Simulation of L-replications with T-samples of process ...")


    X_tvar_2_replications = TensorDict(
        {f"T:{T}": torch.zeros((n_replications, T)) for T in times_T},
        device=device,
        batch_size=[],
    )
    X_dict = TensorDict(
        {f"T:{T}":TensorDict({}, batch_size=[], device=device) for T in times_T},
        batch_size=[],
        device=device,
    )

    for T in times_T:
        for replication in range(n_replications):
            X_tvar_2_T, X = simulation_1_rep_process(d, T) #, device # TDOO
            X_tvar_2_replications[f"T:{T}"][replication] = torch.from_numpy(X_tvar_2_T).to(device) # TODO
            X_dict[f"T:{T}"][str(replication)] = torch.from_numpy(X).to(device) # TODO

    X_tvar_2 = TensorDict(
    {f"t:{t}_T:{T}":torch.empty(n_replications, dtype=torch.float) for t in times_t for T in times_T},
    batch_size=[],
     device=device,
     )
    for t in times_t:
        for T in times_T:
            X_tvar_2[f"t:{t}_T:{T}"] = \
            torch.tensor([X_tvar_2_replications[f"T:{T}"][replication][t-1] for replication in range(n_replications)])

    toc = datetime.now()
    print(f"Simulation completed ; time elapsed = {toc-tic}.")
    return X_tvar_2, X_tvar_2_replications, X_dict


def bandwidth(d, T, lambda_=1.):
    xi = 0.3 / ((d + 1))
    bandwidth = lambda_ * T ** (-xi)
    return bandwidth


def computation_weights(d, lambda_, times_t, times_T, n_replications, X_dict, process, time_kernel, space_kernel, path_dicts,
                        device):
    tic = datetime.now()
    print('-' * 100)
    print(f"Running computation weights starts at {tic} ...")

    gaussian_kernel = {
        f"T:{T}": Kernel(T=T, bandwidth=bandwidth(d, T, lambda_), space_kernel=space_kernel, time_kernel=time_kernel,
                         device=device) for T in times_T
    }

    gaussian_weights = TensorDict(
        {f"t:{t}_T:{T}": TensorDict({}, batch_size=[], device=device) for t in times_t for T in times_T},
        batch_size=[],
        device=device,
    )

    for t in times_t:
        for T in times_T:
            gaussian_weights[f"t:{t}_T:{T}"] = {
                str(replication): gaussian_kernel[f"T:{T}"].fit(X_dict[f"T:{T}"][str(replication)], t - 1) for
                replication in range(n_replications)}

    gaussian_weights_tensor = TensorDict(
        {
            f"t:{times_t[t]}_T:{times_T[T]}": TensorDict({
                str(replication): gaussian_weights[f"t:{times_t[t]}_T:{times_T[T]}"][str(replication)] for replication
                in range(n_replications)
            },
                batch_size=[],
                device=device)
            for t in range(len(times_t)) for T in range(len(times_T))
        },
        batch_size=[],
        device=device,
    )

    dict_name = f"gaussian_weights_tensor_{process}_TimeKernel{time_kernel}_SpaceKernel{space_kernel}_L={n_replications}.pkl"

    with open(os.path.join(path_dicts, dict_name), 'wb') as f:
        pickle.dump(gaussian_weights_tensor, f)

    toc = datetime.now()
    print(f"Weights computation complete at {toc}; time elapsed = {toc - tic}.")
    return gaussian_weights_tensor


def empirical_cdf(times_t, times_T, X_tvar_2, device):
    empirical_cdf_vals = TensorDict(
        {
            f"t:{t}_T:{T}": ECDFTorch(X_tvar_2[f"t:{t}_T:{T}"], device=device).y for t in times_t for T in times_T
        },
        batch_size=[],
        device=device,
    )
    return empirical_cdf_vals


def wasserstein_distances(lambda_, times_t, times_T, n_replications, X_tvar_2_replications, gaussian_weights_tensor,
                          empirical_cdf_vals, process, time_kernel, space_kernel, path_dicts, device, pplot=None):
    tic = datetime.now()
    print('-' * 100)
    print(f"Running wasserstein distances starts at {tic} ...")
    x_rep = TensorDict(
        {
            f"t:{t}_T:{T}": torch.zeros((n_replications, T + 1)) for t in times_t for T in times_T
        },
        batch_size=[],
        device=device,
    )
    y_rep = TensorDict(
        {
            f"t:{t}_T:{T}": torch.zeros((n_replications, T + 1)) for t in times_t for T in times_T
        },
        batch_size=[],
        device=device,
    )
    wasserstein_distances = TensorDict(
        {
            f"t:{t}_T:{T}": TensorDict({}, batch_size=[]) for t in times_t for T in times_T
        },
        batch_size=[],
        device='cpu',
    )

    for replication in range(n_replications):
        for t in times_t:
            for T in times_T:

                weighted_ecdf = ECDFTorch(X_tvar_2_replications[f"T:{T}"][replication],
                                          gaussian_weights_tensor[f"t:{t}_T:{T}"][str(replication)], device=device)

                x_rep[f"t:{t}_T:{str(T)}"][replication] = weighted_ecdf.x
                y_rep[f"t:{t}_T:{str(T)}"][replication] = weighted_ecdf.y

                ecdf = ECDFTorch(X_tvar_2_replications[f"T:{T}"][replication], device=device)

                weighted_ecdf_y = ecdf.y.detach().cpu().numpy()
                ecdf_y = ecdf.y.detach().cpu().numpy()
                distance = wasserstein_distance(weighted_ecdf_y, ecdf_y)
                wasserstein_distances[f"t:{t}_T:{T}"][str(replication)] = distance

                if pplot is not None:
                    x = x_rep[f"t:{t}_T:{str(T)}"][replication]
                    x = x.detach().cpu().numpy()
                    y = y_rep[f"t:{t}_T:{str(T)}"][replication]
                    y = y.detach().cpu().numpy()
                    plt.plot(x, y, label=f"t:{t}_T:{T}")  # _replication:{replication}")
                    plt.xlabel(r'$y$')
                    plt.ylabel(r'$\hat{F}_t(y|x)$')
                    ## plt.xticks(np.arange(0, T+1, 200, dtype=int))
                    ##plt.xlim(-18, 18)
                    plt.title(
                        r'NW CDF estimators, $\hat{F}_{t}(y|{x})=\sum_{a=1}^T\omega_{a}(\frac{t}{T},{x})\mathbf{1}_{Y_{a,T}\leq y}$')
                    plt.legend()
                    plt.tight_layout()
                if pplot is not None:
                    plt.show()
                ##plt.savefig(path_fig+"nadar_watson_weights_", dpi=150)

    wass_distances_empirical_meanNW = {}
    for t in times_t:
        for T in times_T:
            emp_ccf = empirical_cdf_vals[f"t:{t}_T:{T}"].detach().cpu().numpy()
            emp_mean_nw = y_rep[f"t:{t}_T:{T}"].mean(axis=0).detach().cpu().numpy()
            wass_distances_empirical_meanNW[f"t:{t}_T:{T}"] = wasserstein_distance(emp_ccf, emp_mean_nw)

    wass_times_t = {}

    for t in times_t:
        wass_times_t[f"t:{t}"] = []

    for t in times_t:
        for T in times_T:
            wass_times_t[f"t:{t}"].append(wass_distances_empirical_meanNW[f"t:{t}_T:{T}"])

    dict_name = f"wass_times_t_{process}_TimeKernel{time_kernel}_SpaceKernel{space_kernel}_L={n_replications}_lambda_={lambda_}.pkl"
    with open(os.path.join(path_dicts, dict_name), 'wb') as f:
        pickle.dump(wass_times_t, f)

    toc = datetime.now()
    print(f"Wasserstein distances at {toc}; time elapsed = {toc - tic}.")
    return wass_times_t


def plot_results(lambda_, times_t, times_T, n_replications, wass_times_t, process, time_kernel, space_kernel, path_figs,
                 show=False):
    figure_name = f"torch_wassdistance_{process}_TimeKernel{time_kernel}_SpaceKernel{space_kernel}_L={n_replications}_lambda_={lambda_}.png"

    fig = plt.figure(figsize=(8, 4))

    colors = plt.cm.Set1(np.linspace(0, .5, 8))
    markers = ['o', '>', 'D', 'X', "p", 's', 'P', '*']
    for i, t in zip(range(len(times_t)), times_t):
        plt.plot(times_T, wass_times_t[f"t:{t}"], label=f"t:{t}", color=colors[i], marker=markers[i], markersize=6,
                 lw=2)
        plt.xlim(times_T.min(), times_T.max())
        plt.xlabel(r'Sample size ${T}$ ')
        plt.ylabel("Wasserstein distance")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(path_figs, figure_name), dpi=150)
    if show:
        plt.show()


def main():
    if platform.system() == 'Darwin':
        device = torch.device("mps")
    elif platform.system() == 'Linux' or platform.system() == 'Windows':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    device = torch.device("cpu")

    path_figs = '../results/figs'
    path_dicts = '../results/dicts'
    process = 'GaussiantvAR(2)'

    time_kernel = "uniform"
    # space_kernel = "biweight"

    # time_kernel = "uniform"
    # space_kernel = "gaussian"

    space_kernel = "silverman"

    d = 2
    test = True
    lambda_ = 0.25

    times_t, times_T, n_replications = running_test(test, device)

    X_tvar_2, X_tvar_2_replications, X_dict = simulation_L_rep_process(d, times_t, times_T, n_replications, device)

    # exit()

    gaussian_weights_tensor = computation_weights(d, lambda_, times_t, times_T, n_replications, X_dict, process, time_kernel,
                                                  space_kernel, path_dicts, device)

    empirical_cdf_vals = empirical_cdf(times_t, times_T, X_tvar_2, device)

    # times_t = times_t.detach().cpu().numpy()
    # times_T = times_T.detach().cpu().numpy()

    wass_times_t = wasserstein_distances(lambda_, times_t, times_T, n_replications, X_tvar_2_replications,
                                         gaussian_weights_tensor, empirical_cdf_vals, process, time_kernel,
                                         space_kernel, path_dicts, device)

    plot_results(lambda_, times_t, times_T, n_replications, wass_times_t, process, time_kernel, space_kernel, path_figs)


if __name__ == '__main__':
    main()