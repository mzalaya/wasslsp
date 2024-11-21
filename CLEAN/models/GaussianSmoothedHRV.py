#!/usr/bin/env python
# coding: utf-8
# Author: Mokhtar Z. Alaya <alayaelm@utc.fr>

"""
Gaussian Smoothed HRV

"""
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd 
import random
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools
from scipy.stats import mannwhitneyu
import time
import torch
from numba import njit
import pickle
from joblib import Parallel, delayed


from tensordict import TensorDict

import platform
if platform.system() == 'Darwin':
    device = torch.device("mps")
elif platform.system() == 'Linux':
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

import matplotlib.pyplot as plt

params = {'axes.labelsize': 12,
          'font.size': 12,
          'legend.fontsize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 8, # 10
          'text.usetex': True,
          'figure.figsize': (10, 8)}
plt.rcParams.update(params)
import seaborn as sns
from scipy.stats import wasserstein_distance

import sys
sys.path.append('D:/Users/tiniojan/ExperimentsPhD/11-18-24/wasslsp/CLEAN') # Change to your path

from src.torch.utils import *
from src.torch.utils import ECDFTorch
from src.torch.kernels import Kernel

from scipy.stats import wasserstein_distance

path_data = 'D:/Users/tiniojan/ExperimentsPhD/11-18-24/wasslsp/CLEAN/data/'


def running_test(test, device):
    if test is not None:
        df_HRV = pd.read_csv(path_data+'HRVProcessedData[[Beat]].csv')
        data = df_HRV['niHR'].values
        T = data.shape[0]
        times_t = [7950, 8080, 8150, 8390, 8500, 8650, 8900, 8960]
        times_sigma = [1, 1e-1, 1e-2, 1e-3, 1e-4]
        n_replications = 10
        iterations = 1
    else:
        T = 100
        times_t = [10, 20]
        times_sigma = [1]
        n_replications = 1
        iterations = 1

    print(
        f'times_t [{times_t}] | \n'
        f'T = {T} | \n'
        f'n_replications = {n_replications} | device = {device}'
    )

    return data, times_t, T, n_replications, times_sigma, iterations




@njit
def simulation_1_rep_real(data, d, T, sigma):
    t = 1
    epsilon = np.random.normal(0., sigma, (T,))  
    X = np.zeros((T, d))
    real_T = np.zeros(T)
    while t <= T - 1:
        real_T[t] = data[t] + epsilon[t]  
        X[t] = np.array([real_T[t - 1]])
        t += 1
    return real_T, X


def simulation_L_rep_real(data, T, d, times_t, n_replications, device, output_dir, iterations, times_sigma):
    tic = datetime.now()
    print('-' * 100)
    print(f"Simulation of L-replications of {data} ...")

    # Initialize variables 
    X_real_iteration = None
    X_real_replications = {f"sigma:{sigma}": torch.zeros((n_replications, T), dtype=torch.float16).to(device) for sigma in times_sigma}
    X_dict = {f"sigma:{sigma}": {} for sigma in times_sigma}

    for iteration in range(iterations):
        # Initialize or reset variables for each iteration
        X_real_replications = {f"sigma:{sigma}": torch.zeros((n_replications, T), dtype=torch.float16).to(device) for sigma in times_sigma}
        X_dict = {f"sigma:{sigma}": {} for sigma in times_sigma}
        X_real = TensorDict(
            {f"t:{t}_sigma:{sigma}": torch.empty(n_replications, dtype=torch.float16)
             for sigma in times_sigma for t in times_t},
            device=device
        )

        for sigma in times_sigma:
            for replication in range(n_replications):
                X_real_T, X = simulation_1_rep_real(data, d, T, sigma)
                X_real_replications[f"sigma:{sigma}"][replication] = torch.from_numpy(X_real_T).to(device)
                X_dict[f"sigma:{sigma}"][str(replication)] = torch.from_numpy(X).to(device)
    
        for sigma in times_sigma:
            for t in times_t:
                # Extract data using list comprehension and avoid creating intermediate tensors
                data = [X_real_replications[f"sigma:{sigma}"][replication][t - 1].item()
                        for replication in range(n_replications)]
                X_real[f"t:{t}_sigma:{sigma}"][:] = torch.tensor(data, dtype=torch.float16, device=device)

        # Save the current iteration's results to disk using pickle
        iteration_results = {
            "X_real_replications": X_real_replications,
            "X_dict": X_dict,
            "X_real_iteration": X_real  
        }
        with open(os.path.join(output_dir, f"iteration_{iteration + 1}.pkl"), "wb") as f:
            pickle.dump(iteration_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Keep the last iteration's X_real for returning
        X_real_iteration = X_real

        # Release memory after saving
        del X_real_replications, X_dict, iteration_results
        torch.cuda.empty_cache()  # Use if running on GPU

        print(f"Montecarlo {iteration + 1} completed.")

    toc = datetime.now()
    print(f"Simulation completed; time elapsed = {toc - tic}.")
    
    # Ensure that variables are returned correctly
    if X_real_iteration is not None:
        return X_real_iteration, {f"sigma:{sigma}": torch.zeros((n_replications, T), dtype=torch.float16).to(device) for sigma in times_sigma}, {f"sigma:{sigma}": {} for sigma in times_sigma}
    else:
        raise ValueError("X_real was not properly assigned during the iterations.")


def bandwidth(d, T, lambda_=1.):
    xi = 0.2 / ((d + 1))
    bandwidth = lambda_ * T ** (-xi)
    return bandwidth


def computation_weights(T, d, lambda_, times_t, times_sigma, iterations, n_replications, X_dict, 
                        time_kernel, space_kernel, input_dir, output_dir, device):
    
    for iteration in range(iterations):
        tic = datetime.now()
        print('-' * 100)
        print(f"Running computation weights for Montecarlo {iteration + 1} starts at {tic} ...")
        
        gaussian_kernel = TensorDict(
            {f"sigma:{sigma}": Kernel(T=T, bandwidth=bandwidth(d, T, lambda_), space_kernel=space_kernel, time_kernel=time_kernel, device=device)
             for sigma in times_sigma},
            device=device
        )

        gaussian_weights = TensorDict(
            {f"t:{t}_sigma:{sigma}": TensorDict({}, device=device)
             for sigma in times_sigma for t in times_t},
            device=device
        )

        # Load data from pickle files
        with open(os.path.join(input_dir, f"iteration_{iteration + 1}.pkl"), "rb") as f:
            iteration_results = pickle.load(f)
    
        X_dict = iteration_results["X_dict"]

        for sigma in times_sigma:
            for t in times_t:
                gaussian_weights[f"t:{t}_sigma:{sigma}"] = {
                    str(replication): gaussian_kernel[f"sigma:{sigma}"].fit(
                        X_dict[f"sigma:{sigma}"][str(replication)], t - 1
                    ) for replication in range(n_replications)
                }
    
        # Save the gaussian_weights for the current iteration to disk using pickle
        with open(os.path.join(output_dir, f"gaussian_weights_iteration_{iteration + 1}.pkl"), "wb") as f:
            pickle.dump(gaussian_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
    
        # Release memory after saving
        del gaussian_kernel, X_dict, iteration_results
        torch.cuda.empty_cache()  # Clear GPU cache if using CUDA

        toc = datetime.now()
        print(f"Weights computation complete for Montecarlo {iteration + 1} at {toc}; time elapsed = {toc - tic}.")

    # Return the result after all iterations are complete
    return gaussian_weights


def empirical_cdf(times_t, times_sigma, device, iterations, input_dir, X_real):
    empirical_cdfs_iterations = {}

    for iteration in range(iterations):

        with open(os.path.join(input_dir, f"iteration_{iteration + 1}.pkl"), "rb") as f:
            iteration_results = pickle.load(f)

        X_real = iteration_results["X_real_iteration"]

        empirical_cdfs = TensorDict(
            {
                f"t:{t}_sigma:{sigma}": ECDFTorch(X_real[f"t:{t}_sigma:{sigma}"]).y
                for sigma in times_sigma for t in times_t  
            },
            device=device,
        )
    
        empirical_cdfs_iterations[f"Iteration:{iteration + 1}"] = empirical_cdfs
    
    return empirical_cdfs_iterations


def wasserstein_distances(T, times_t, times_sigma, n_replications, iterations, gaussian_weights, empirical_cdfs_iterations, X_real_replications, 
                          input_dir, input_dir_weights, output_dir, device, pplot=None):
    
    wass_distances_empirical_meanNW_iterations = {}

    for iteration in range(iterations):
        tic = datetime.now()
        print('-' * 100)
        print(f"Running wasserstein distances starts at {tic} ...")

        # Load gaussian_weights_tensor from disk for the current iteration
        with open(os.path.join(input_dir_weights, f"gaussian_weights_iteration_{iteration + 1}.pkl"), "rb") as f:
            gaussian_weights = pickle.load(f)

        # Load the iteration results for the current iteration to access X_real_replications
        with open(os.path.join(input_dir, f"iteration_{iteration + 1}.pkl"), "rb") as f:
            iteration_results = pickle.load(f)
    
        X_real_replications = iteration_results["X_real_replications"]

        # Initialize x_rep and y_rep TensorDicts for the current iteration
        x_rep = TensorDict(
            {
                f"t:{t}_sigma:{sigma}": torch.zeros((n_replications, T + 1), dtype=torch.float16)
                for sigma in times_sigma for t in times_t
            },
            device=device,
        )

        y_rep = TensorDict(
            {
                f"t:{t}_sigma:{sigma}": torch.zeros((n_replications, T + 1), dtype=torch.float16)
                for sigma in times_sigma for t in times_t
            },
            device=device,
        )

        for replication in range(n_replications):
            for sigma in times_sigma:
                for t in times_t:
                    # Calculate the weighted ECDF using data loaded from disk
                    weighted_ecdf = ECDFTorch(
                        X_real_replications[f"sigma:{sigma}"][replication],
                        gaussian_weights[f"t:{t}_sigma:{sigma}"][str(replication)]
                    )

                    x_rep[f"t:{t}_sigma:{sigma}"][replication] = weighted_ecdf.x
                    y_rep[f"t:{t}_sigma:{sigma}"][replication] = weighted_ecdf.y

                    # Optional plotting
                    if pplot is not None:
                        x = x_rep[f"t:{t}_sigma:{sigma}"][replication].detach().cpu().numpy()
                        y = y_rep[f"t:{t}_sigma:{sigma}"][replication].detach().cpu().numpy()
                        plt.plot(x, y, label=f"t:{t}_sigma:{sigma}")
                        plt.xlabel(r'$y$')
                        plt.ylabel(r'$\hat{F}_t(y|x)$')
                        plt.title(r'NW CDF estimators, $\hat{F}_{t}(y|{x})=\sum_{a=1}^T\omega_{a}(\frac{t}{T},{x})\mathbf{1}_{Y_{a,T}\leq y}$')
                        plt.legend()
                        plt.tight_layout()

                    if pplot is not None:
                        plt.show()
        
        iteration_empirical_cdfs = empirical_cdfs_iterations[f"Iteration:{iteration + 1}"]

        wass_distances_empirical_meanNW = {}

        for sigma in times_sigma:
            for t in times_t:
                emp_ccf = iteration_empirical_cdfs[f"t:{t}_sigma:{sigma}"].detach().cpu().numpy()
                emp_mean_nw = y_rep[f"t:{t}_sigma:{sigma}"].mean(axis=0).detach().cpu().numpy()
                wass_distances_empirical_meanNW[f"t:{t}_sigma:{sigma}"] = wasserstein_distance(emp_ccf, emp_mean_nw)
    
        wass_distances_empirical_meanNW_iterations[f"Iteration:{iteration + 1}"] = wass_distances_empirical_meanNW

        # Save the results to disk using pickle
        with open(f"{output_dir}/wass_distances_empirical_meanNW_iterations.pkl", "wb") as f:
            pickle.dump(wass_distances_empirical_meanNW_iterations, f)

        toc = datetime.now()
        print(f"Wasserstein distances at {toc}; time elapsed = {toc - tic}.")
        print(wass_distances_empirical_meanNW_iterations)
    
    return wass_distances_empirical_meanNW_iterations


def plot_results(times_t, times_sigma, n_replications, input_dir, T, iterations, wass_distances_empirical_meanNW_iterations):

    sns.set(style="darkgrid")
    plt.rcParams['text.usetex'] = False
    plt.rcParams["figure.figsize"] = (6, 4)
    colorlist = ["light orange", "dark orange", "salmon pink", "neon pink", "cornflower", "cobalt blue",
             "blue green", "aquamarine", "dark orange", "golden yellow", "reddish pink", "black", "reddish purple"]
    colors = sns.xkcd_palette(colorlist)
    markers = ['o', 'p', 's', 'd', 'h', 'o', 'p', '<', '>', '8', 'P']
    linestylev = ['-', '--', ':', '-', '--', ':']
    rep = n_replications  # Use `n_replications` for consistency with previous context

    # Load the pickle file containing Wasserstein distances
    with open(f"{input_dir}/wass_distances_empirical_meanNW_iterations.pkl", "rb") as f: 
        wass_distances_empirical_meanNW_iterations = pickle.load(f)

    # Calculate t/T and format to 3 decimals for categories
    times_t_T = [round(t / T, 3) for t in times_t]
    times_t_T_cat = [f"{t:.3f}" for t in times_t_T]  # Category labels

    # Create a single plot with all sigma values
    for i, sigma in enumerate(times_sigma):
        distances = [
            wass_distances_empirical_meanNW_iterations['Iteration:1'].get(
                f"t:{t}_sigma:{sigma}", 0
            )
            for t in times_t
        ]
    
        # Construct sigma label
        exponent = int(np.log10(sigma))
        if exponent == 0:
            sigma_label = r"$\sigma$: 1"
        else:
            sigma_label = rf"$\sigma$: $10^{{{exponent}}}$"

        plt.plot(
            times_t_T_cat, 
            distances, 
            label=sigma_label, 
            color=colors[i % len(colors)], 
            marker=markers[i % len(markers)], 
            markersize=6, 
            lw=2,
            linestyle=linestylev[i % len(linestylev)]
        )

    plt.xlabel(r'$t/T$', fontsize=16)
    plt.ylabel("Wasserstein distance", fontsize=16)
    plt.xticks(times_t_T_cat, fontsize=16)  # Set x-axis as categorical labels
    plt.yticks(fontsize=16)
    plt.grid(True, axis='x')  # Grid lines for x-axis
    plt.legend(fontsize=15, loc='best')
    plt.tight_layout()
    plt.savefig(f"W1_BabyECG_all_sigmas_rep-{rep}.pdf", dpi=150)
    plt.show()


def replication_hrv(data=None, sigma=1., d=1, n_replications=1):
    """
    Replication generating procedure of HRV data
    """
    if data is None or data.shape[0] == 0:
        raise ValueError("Input data must be a non-empty NumPy array.")

    T = data.shape[0]
    real_replications = []

    for replication in range(n_replications):
        np.random.seed(31)
        t = 1
        epsilon = np.random.normal(size=(T), scale=sigma)
        real_ = np.zeros(T)
        
        while t <= T - 1:
            real_[t] = data[t] + epsilon[t]
            t += 1
        
        real_replications.append(real_)
        
    return real_replications


def plot_sample_noised(data, T, sigma_fix, n_replications_fix):
    """
    Plot replications of HRV data with added noise
    """
    sns.set(style="darkgrid")
    plt.rcParams['text.usetex'] = False
    plt.rcParams["figure.figsize"] = (10, 3)

    # Define colors and other styles
    colorlist = ["neon pink", "cobalt blue", "aquamarine"]
    colors = sns.xkcd_palette(colorlist)

    # Generate replications
    X_real_replications = replication_hrv(data, sigma=sigma_fix, n_replications=n_replications_fix)

    for replication in range(n_replications_fix):
        # Validate replication data
        if len(X_real_replications[replication]) == 0:
            raise ValueError(f"Replication {replication} is empty!")

        # Plot the replication
        plt.plot(
            X_real_replications[replication],
            lw=.5,
            label=f'Replication n°{replication}',
            color=colors[replication % len(colors)]
        )

        # Dynamic plot limits
        plt.ylim(
            min(X_real_replications[replication][1:]),
            max(X_real_replications[replication])
        )
        plt.xlim(0, T)

    # Final plot adjustments
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{n_replications_fix} replications of HRV sigma={sigma_fix}.pdf", dpi=300)
    plt.show()


def plot_nw_cdf_estimators(x_rep, y_rep, t_fixed, sigma_fix, n_replications_fix, T, iterations, input_dir_weights, input_dir):
    
    sns.set(style="darkgrid")
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(6, 4))
    colorlist = ["neon pink", "cobalt blue", "aquamarine"]
    colors = sns.xkcd_palette(colorlist)

    for iteration in range(iterations):
        
        # Load gaussian_weights_tensor from disk for the current iteration
        with open(os.path.join(input_dir_weights, f"gaussian_weights_iteration_{iteration + 1}.pkl"), "rb") as f:
            gaussian_weights = pickle.load(f)

        # Load the iteration results for the current iteration to access X_real_replications
        with open(os.path.join(input_dir, f"iteration_{iteration + 1}.pkl"), "rb") as f:
            iteration_results = pickle.load(f)
    
        X_real_replications = iteration_results["X_real_replications"]

        # Initialize x_rep and y_rep TensorDicts for the current iteration
        x_rep = TensorDict(
            {
                f"t:{t}_sigma:{sigma}": torch.zeros((n_replications_fix, T + 1), dtype=torch.float16)
                for sigma in sigma_fix for t in t_fixed
            },
            device=device,
        )

        y_rep = TensorDict(
            {
                f"t:{t}_sigma:{sigma}": torch.zeros((n_replications_fix, T + 1), dtype=torch.float16)
                for sigma in sigma_fix for t in t_fixed
            },
            device=device,
        )

        for replication in range(n_replications_fix):
            for sigma in sigma_fix:
                for t in t_fixed:
                    # Calculate the weighted ECDF using data loaded from disk
                    weighted_ecdf = ECDFTorch(
                        X_real_replications[f"sigma:{sigma}"][replication],
                        gaussian_weights[f"t:{t}_sigma:{sigma}"][str(replication)]
                    )

                    x_rep[f"t:{t}_sigma:{sigma}"][replication] = weighted_ecdf.x
                    y_rep[f"t:{t}_sigma:{sigma}"][replication] = weighted_ecdf.y

                    
                    x = x_rep[f"t:{t}_sigma:{sigma}"][replication].detach().cpu().numpy()
                    y = y_rep[f"t:{t}_sigma:{sigma}"][replication].detach().cpu().numpy()
                    plt.plot(x, y, label=f"replication: {replication}", lw=2, color = colors[replication % len(colors)])
                    plt.xlabel("HRV values", fontsize=16)
                    plt.ylabel("NW Conditional CDF", fontsize = 16)
                    plt.xticks(fontsize=16)
                    plt.yticks(fontsize=16)
                    plt.grid(True)
                    plt.legend(fontsize=15)
        
        plt.tight_layout()
        plt.savefig(f"{n_replications_fix} NW estimators of HRV sigma={sigma_fix}.pdf", dpi=300)
        plt.show()


def main():
    if platform.system() == 'Darwin':
        device = torch.device("mps")
    elif platform.system() == 'Linux' or platform.system() == 'Windows':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    device = torch.device("cpu")

    ### Generating L replications of Gaussian-smoothed dataset
    output_dir = "simulation_results"
    os.makedirs(output_dir, exist_ok=True)

    d = 1
    test = True
    lambda_ = 1.

    data, times_t, T, n_replications, times_sigma, iterations = running_test(test, device)
    X_real, X_real_replications, X_dict = simulation_L_rep_real(data, T, d, times_t, n_replications, device, output_dir, iterations, times_sigma)

    ### Weights calculation for NW Conditional CDFs
    input_dir = "simulation_results"
    output_dir = "gaussian_weights_output"
    os.makedirs(output_dir, exist_ok=True)

    time_kernel = "uniform"
    space_kernel = "gaussian"
    gaussian_weights = computation_weights(T, d, lambda_, times_t, times_sigma, iterations, n_replications, X_dict, 
                        time_kernel, space_kernel, input_dir, output_dir, device)
    
    ### Empirical CDF calculation
    input_dir = "simulation_results"
    empirical_cdfs_iterations = empirical_cdf(times_t, times_sigma, device, iterations, input_dir, X_real)

    ### Wasserstein distances between Average NW Conditional CDF and Empirical CDF
    input_dir = "simulation_results"
    input_dir_weights = "gaussian_weights_output"
    output_dir = "BabyECG_Wass"
    os.makedirs(output_dir, exist_ok=True)

    wass_distances_empirical_meanNW_iterations = wasserstein_distances(T, times_t, times_sigma, n_replications, iterations, 
                                                                       gaussian_weights, empirical_cdfs_iterations, X_real_replications, 
                                                                       input_dir, input_dir_weights, output_dir, device, pplot=None)
    
    
    ### Plot 3 sample Gaussian-smoothed data
    sigma_fix = 1.0
    n_replications_fix = 3
    plot_sample_noised(data, T, sigma_fix, n_replications_fix)


    ### Plot 3 Sample NW CDF Estimators at Fixed t and Fixed Sigma
    input_dir = "simulation_results"
    input_dir_weights = "gaussian_weights_output"
    t_fixed = [7950]  
    sigma_fix = [1]  
    x_rep = {}
    y_rep = {}
    plot_nw_cdf_estimators(x_rep, y_rep, t_fixed, sigma_fix, n_replications_fix, T, iterations, input_dir_weights, input_dir)

    ### Plot of Wasserstein distances at different t/T
    input_dir = "BabyECG_Wass"
    plot_results(times_t, times_sigma, n_replications, input_dir, T, iterations, wass_distances_empirical_meanNW_iterations)




    


if __name__ == '__main__':
    main()