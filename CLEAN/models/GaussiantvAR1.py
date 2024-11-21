#!/usr/bin/env python
# coding: utf-8
# Author: Mokhtar Z. Alaya <alayaelm@utc.fr>


"""
Gaussian tvAR(1)
Ref: S. Richter and R. Dahlhaus. Cross validation for locally stationary processes. Ann. Statist., 47(4):2145–2173, 2019
"""
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')
from numba import njit
from numba import cuda
import pickle
from joblib import Parallel, delayed

import time
import numpy as np
import torch
import seaborn as sns

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
sys.path.append('D:/Users/tiniojan/ExperimentsPhD/11-18-24/wasslsp/CLEAN') # Change to your path

from src.torch.utils import *
from src.torch.kernels import Kernel

from scipy.stats import wasserstein_distance



def running_test(test, device):
    if test is not None:
        times_T = [5000, 10000, 15000]
        times_t_dict = {
            5000: [2400, 2410, 2430, 2440, 2460, 2470, 2490, 2500, 2510, 2530, 2540, 2560, 2570, 2590, 2600],
            10000: [4800, 4830, 4860, 4890, 4910, 4940, 4970, 5000, 5030, 5060, 5090, 5110, 5140, 5170, 5200],
            15000: [7200, 7240, 7290, 7330, 7370, 7410, 7460, 7500, 7540, 7580, 7630, 7670, 7710, 7760, 7800],
            }
        n_replications = 1000
        iterations = 100

    else:
        times_T = [5, 10]
    
        times_t_dict = {
            5: [1, 2],
            10: [3, 4]
        }
        n_replications = 2
        iterations = 1

    for T in times_T:
        times_t = times_t_dict[T]
        print(
        f'times_T = {T} | \n'
        f'times_t_dict [{times_t}] | \n'
        f'n_replications = {n_replications} | device = {device}'
        )

    return times_t_dict, times_T, n_replications, iterations



@njit
def phi_star(u):
    return 0.9 * np.sin(2 * np.pi * u)

@njit
def m_star(u, x):
    return phi_star(u) * x


def plot_m_star(process, T_samples):
    T_samples = 1000
    u = np.linspace(0., 1., T_samples) 
    x = np.linspace(-5, 5, T_samples)

    m_star_vals = np.array([m_star(u_val, x_val) for u_val, x_val in zip(u, x)])

    plt.rcParams['text.usetex'] = False
    plt.rcParams["figure.figsize"] = (10,3)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.minorticks_on()
    ax.scatter(u, x, m_star_vals, c=m_star_vals, cmap='viridis', linewidth=0.1)
    ax.set_xlabel(r'$u$', fontsize=16)
    ax.set_ylabel(r'$x$', fontsize=16)
    ax.set_title(r'$m^\star(u,x)$', fontsize=16)
    figure_name_mean_function = f"Conditional_mean_function_of_process_{process}.pdf"
    plt.tight_layout() 
    fig.savefig(os.path.join(figure_name_mean_function), dpi=150)
    plt.show()


@njit
def simulation_1_rep_process(d, T):
    t = 1
    epsilon = np.random.normal(0., 1., (T,))
    X = np.zeros((T, d))
    X_tvar_1_T = np.zeros(T)
    while t <= T - 1:
        m_star_val = m_star(t / T, X_tvar_1_T[t - 1])
        X_tvar_1_T[t] = m_star_val + epsilon[t]  
        X[t] = np.array([X_tvar_1_T[t - 1]])
        t += 1
    return X_tvar_1_T, X


def plot_1_rep_process(process, d, T_samples, device):
    
    sns.set(style="darkgrid")
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(10, 3))

    # Initialize variables
    X_tvar_1_replications = torch.zeros((1, T_samples)).to(device)  # For 1 replication
    X_dict = {}

    torch.manual_seed(31)  
    X = torch.zeros((T_samples, d)).to(device)  
    X_tvar_1_np = np.zeros(T_samples)  

    X_tvar_1_np, X = simulation_1_rep_process(d, T_samples)

    # Save results
    X_tvar_1 = torch.tensor(X_tvar_1_np).to(device)
    X_dict["0"] = X
    X_tvar_1_replications[0] = X_tvar_1

    # Plotting
    fig = plt.figure(figsize=(10, 3))
    plt.grid(which="major", color="#DDDDDD", linewidth=0.8)
    plt.minorticks_on()
    plt.plot(X_tvar_1_replications[0].detach().cpu().numpy(), lw=1, label="Replication 1")
    plt.xlim(0, T_samples)
    plt.ylim(
        min(X_tvar_1_replications[0].detach().cpu().numpy()),
        max(X_tvar_1_replications[0].detach().cpu().numpy())
    )
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    figure_name_process = f"Simulation_of_{process}_with_Time_Samples_{T_samples}.pdf"
    fig.savefig(figure_name_process, dpi=150)
    plt.show()


def simulation_L_rep_process(d, times_t_dict, times_T, n_replications, device, output_dir, iterations):
    tic = datetime.now()
    print('-' * 100)
    print("Simulation of L-replications with T-samples of process ...")

    # Initialize variables 
    X_tvar_1_iteration = None
    X_tvar_1_replications = {f"T:{T}": torch.zeros((n_replications, T), dtype=torch.float16).to(device) for T in times_T}
    X_dict = {f"T:{T}": {} for T in times_T}

    for iteration in range(iterations):
        # Initialize or reset variables for each iteration
        X_tvar_1_replications = {f"T:{T}": torch.zeros((n_replications, T), dtype=torch.float16).to(device) for T in times_T}
        X_dict = {f"T:{T}": {} for T in times_T}
        X_tvar_1 = TensorDict(
            {f"t:{t}_T:{T}": torch.empty(n_replications, dtype=torch.float16)
             for T in times_T for t in times_t_dict[T]},
            device=device
        )

        for T in times_T:
            for replication in range(n_replications):
                X_tvar_1_T, X = simulation_1_rep_process(d, T)
                X_tvar_1_replications[f"T:{T}"][replication] = torch.from_numpy(X_tvar_1_T).to(device)
                X_dict[f"T:{T}"][str(replication)] = torch.from_numpy(X).to(device)
    
        for T in times_T:
            times_t = times_t_dict[T]
            for t in times_t:
                # Extract data using list comprehension and avoid creating intermediate tensors
                data = [X_tvar_1_replications[f"T:{T}"][replication][t - 1].item()
                        for replication in range(n_replications)]
                X_tvar_1[f"t:{t}_T:{T}"][:] = torch.tensor(data, dtype=torch.float16, device=device)

        # Save the current iteration's results to disk using pickle
        iteration_results = {
            "X_tvar_1_replications": X_tvar_1_replications,
            "X_dict": X_dict,
            "X_tvar_1_iteration": X_tvar_1  
        }
        with open(os.path.join(output_dir, f"iteration_{iteration + 1}.pkl"), "wb") as f:
            pickle.dump(iteration_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Keep the last iteration's X_tvar_1 for returning
        X_tvar_1_iteration = X_tvar_1

        # Release memory after saving
        del X_tvar_1_replications, X_dict, iteration_results
        torch.cuda.empty_cache()  # Use if running on GPU

        print(f"Montecarlo {iteration + 1} completed.")

    toc = datetime.now()
    print(f"Simulation completed; time elapsed = {toc - tic}.")
    
    # Ensure that variables are returned correctly
    if X_tvar_1_iteration is not None:
        return X_tvar_1_iteration, {f"T:{T}": torch.zeros((n_replications, T), dtype=torch.float16).to(device) for T in times_T}, {f"T:{T}": {} for T in times_T}
    else:
        raise ValueError("X_tvar_1 was not properly assigned during the iterations.")


def bandwidth(d, T, lambda_=1.):
    xi = 0.2 / ((d + 1))
    bandwidth = lambda_ * T ** (-xi)
    return bandwidth


def computation_weights(d, lambda_, times_t_dict, times_T, iterations, n_replications, X_dict, 
                        time_kernel, space_kernel, input_dir, output_dir, device):
    
    for iteration in range(iterations):
        tic = datetime.now()
        print('-' * 100)
        print(f"Running computation weights for Montecarlo {iteration + 1} starts at {tic} ...")
        
        gaussian_kernel = TensorDict(
            {f"T:{T}": Kernel(T=T, bandwidth=bandwidth(d, T, lambda_), space_kernel=space_kernel, time_kernel=time_kernel, device=device)
             for T in times_T},
            device=device
        )

        gaussian_weights = TensorDict(
            {f"t:{t}_T:{T}": TensorDict({}, device=device)
             for T in times_T for t in times_t_dict[T]},
            device=device
        )

        # Load data from pickle files
        with open(os.path.join(input_dir, f"iteration_{iteration + 1}.pkl"), "rb") as f:
            iteration_results = pickle.load(f)
    
        X_dict = iteration_results["X_dict"]

        for T in times_T:
            times_t = times_t_dict[T]
            for t in times_t:
                gaussian_weights[f"t:{t}_T:{T}"] = {
                    str(replication): gaussian_kernel[f"T:{T}"].fit(
                        X_dict[f"T:{T}"][str(replication)], t - 1
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


def empirical_cdf(times_t_dict, times_T, device, iterations, input_dir, X_tvar_1):
    empirical_cdfs_iterations = {}

    for iteration in range(iterations):

        with open(os.path.join(input_dir, f"iteration_{iteration + 1}.pkl"), "rb") as f:
            iteration_results = pickle.load(f)

        X_tvar_1 = iteration_results["X_tvar_1_iteration"]

        empirical_cdfs = TensorDict(
            {
                f"t:{t}_T:{T}": ECDFTorch(X_tvar_1[f"t:{t}_T:{T}"]).y
                for T in times_T for t in times_t_dict[T]  
            },
            device=device,
        )
    
        empirical_cdfs_iterations[f"Iteration:{iteration + 1}"] = empirical_cdfs
    
    return empirical_cdfs_iterations


def wasserstein_distances(times_t_dict, times_T, n_replications, iterations, gaussian_weights, empirical_cdfs_iterations, X_tvar_1_replications, 
                          input_dir, input_dir_weights, output_dir, device, pplot=None):
    
    wass_distances_empirical_meanNW_iterations = {}

    for iteration in range(iterations):
        tic = datetime.now()
        print('-' * 100)
        print(f"Running wasserstein distances starts at {tic} ...")

        # Load gaussian_weights_tensor from disk for the current iteration
        with open(os.path.join(input_dir_weights, f"gaussian_weights_iteration_{iteration + 1}.pkl"), "rb") as f:
            gaussian_weights = pickle.load(f)

        # Load the iteration results for the current iteration to access X_tvar_1_replications
        with open(os.path.join(input_dir, f"iteration_{iteration + 1}.pkl"), "rb") as f:
            iteration_results = pickle.load(f)
    
        X_tvar_1_replications = iteration_results["X_tvar_1_replications"]

        # Initialize x_rep and y_rep TensorDicts for the current iteration
        x_rep = TensorDict(
            {
                f"t:{t}_T:{T}": torch.zeros((n_replications, T + 1), dtype=torch.float16)
                for T in times_T for t in times_t_dict[T]
            },
            device=device,
        )

        y_rep = TensorDict(
            {
                f"t:{t}_T:{T}": torch.zeros((n_replications, T + 1), dtype=torch.float16)
                for T in times_T for t in times_t_dict[T]
            },
            device=device,
        )

        for replication in range(n_replications):
            for T in times_T:
                times_t = times_t_dict[T]
                for t in times_t:
                    # Calculate the weighted ECDF using data loaded from disk
                    weighted_ecdf = ECDFTorch(
                        X_tvar_1_replications[f"T:{T}"][replication],
                        gaussian_weights[f"t:{t}_T:{T}"][str(replication)]
                    )

                    x_rep[f"t:{t}_T:{T}"][replication] = weighted_ecdf.x
                    y_rep[f"t:{t}_T:{T}"][replication] = weighted_ecdf.y

                    # Optional plotting
                    if pplot is not None:
                        x = x_rep[f"t:{t}_T:{T}"][replication].detach().cpu().numpy()
                        y = y_rep[f"t:{t}_T:{T}"][replication].detach().cpu().numpy()
                        plt.plot(x, y, label=f"t:{t}_T:{T}")
                        plt.xlabel(r'$y$')
                        plt.ylabel(r'$\hat{F}_t(y|x)$')
                        plt.title(r'NW CDF estimators, $\hat{F}_{t}(y|{x})=\sum_{a=1}^T\omega_{a}(\frac{t}{T},{x})\mathbf{1}_{Y_{a,T}\leq y}$')
                        plt.legend()
                        plt.tight_layout()

                    if pplot is not None:
                        plt.show()
        
        iteration_empirical_cdfs = empirical_cdfs_iterations[f"Iteration:{iteration + 1}"]

        wass_distances_empirical_meanNW = {}

        for T in times_T:
            times_t = times_t_dict[T]  
            for t in times_t:
                emp_ccf = iteration_empirical_cdfs[f"t:{t}_T:{T}"].detach().cpu().numpy()
                emp_mean_nw = y_rep[f"t:{t}_T:{T}"].mean(axis=0).detach().cpu().numpy()
                wass_distances_empirical_meanNW[f"t:{t}_T:{T}"] = wasserstein_distance(emp_ccf, emp_mean_nw)
    
        wass_distances_empirical_meanNW_iterations[f"Iteration:{iteration + 1}"] = wass_distances_empirical_meanNW

        # Save the results to disk using pickle
        with open(f"{output_dir}/wass_distances_empirical_meanNW_iterations.pkl", "wb") as f:
            pickle.dump(wass_distances_empirical_meanNW_iterations, f)

        toc = datetime.now()
        print(f"Wasserstein distances at {toc}; time elapsed = {toc - tic}.")
    
    return wass_distances_empirical_meanNW_iterations
    
def wass_stats(input_dir, output_dir, times_T, times_t_dict, iterations, wass_distances_empirical_meanNW_iterations):
    with open(os.path.join(f"{input_dir}/wass_distances_empirical_meanNW_iterations.pkl"), "rb") as f:
        wass_distances_empirical_meanNW_iterations = pickle.load(f)

    wass_distances_stats = {}

    for T in times_T:
        times_t = times_t_dict[T]  
        for t in times_t:
            distances_all_iterations = []

            for iteration in range(iterations):
                iteration_wass_distances = wass_distances_empirical_meanNW_iterations[f"Iteration:{iteration + 1}"]
                distances_all_iterations.append(iteration_wass_distances[f"t:{t}_T:{T}"])

            if distances_all_iterations:
                distances_array = np.array(distances_all_iterations)
                mean_distance = distances_array.mean()
                std_distance = distances_array.std()
            else:
                mean_distance = None
                std_distance = None

            wass_distances_stats[f"t:{t}_T:{T}"] = {
                "mean": mean_distance,
                "std": std_distance
            }

    # Print the calculated mean and standard deviation for verification

    for key, stats in wass_distances_stats.items():
        mean = stats['mean']
        std = stats['std']
        if mean is not None and std is not None:
            print(f"{key} -> Mean: {mean:.6f}, Std: {std:.6f}")
        else:
            print(f"{key} -> Insufficient data for mean and std calculation.")

    # Save the new stats to a pickle file in the specified folder
    with open(f"{output_dir}/wass_distances_stats.pkl", "wb") as f:
        pickle.dump(wass_distances_stats, f)

    return wass_distances_stats

def plot_results(times_t_dict, times_T, n_replications, iterations, process, wass_distances_stats, time_kernel, space_kernel, input_dir):
    with open(os.path.join(f"{input_dir}/wass_distances_stats.pkl"), "rb") as f:
        wass_distances_stats = pickle.load(f)

    plt.rcParams['text.usetex'] = False
    sns.set(style="darkgrid")
    plt.figure(figsize=(8, 4))
    colorlist = ["cobalt blue", "neon pink", "golden yellow"]
    colors = sns.xkcd_palette(colorlist)
    markert = ['o','p','s','d','h','o','p','<','>','8','P']
    linestylev = ['-','--',':']


    for i, T in enumerate(times_T):
        times_t = times_t_dict[T]  
    
        mean_distances = []
        std_distances = []
        normalized_times_t = [t / T for t in times_t]  
    
        for t in times_t:
            key = f"t:{t}_T:{T}"
            if key in wass_distances_stats:
                mean = wass_distances_stats[key]['mean']
                std = wass_distances_stats[key]['std']
            
                if mean is not None and std is not None:
                    mean_distances.append(mean)
                    std_distances.append(std)
    
        mean_distances = np.array(mean_distances)
        std_distances = np.array(std_distances)
    
        color = colors[i % len(colors)]
    
        plt.plot(normalized_times_t, 
            mean_distances, 
            lw = 2, 
            marker = markert[i], 
            markersize=10,
            c=colors[i],
            linestyle=linestylev[i],
            label=f"T={T}"
            )
    
        plt.fill_between(
            normalized_times_t, 
            mean_distances - std_distances, 
            mean_distances + std_distances, 
            color=color, 
            alpha=0.2, 
            #label=f"± Std Dev"
        )

    plt.xlabel(r'$t/T$', fontsize=16)
    plt.xlim(0.48, 0.52)
    plt.xticks(np.linspace(0.48, 0.52, 9), fontsize=16)
    plt.ylabel('Expected Wasserstein Distance', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(f"EW1_{process}_timekernel-{time_kernel}_spacekernel-{space_kernel}_rep-{n_replications}_iter-{iterations}.pdf", dpi=200)
    plt.show()

def main():
    if platform.system() == 'Darwin':
        device = torch.device("mps")
    elif platform.system() == 'Linux' or platform.system() == 'Windows':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    device = torch.device("cpu")

    ### Generating L replications of LSP
    output_dir = "simulation_results"
    os.makedirs(output_dir, exist_ok=True)

    d = 1
    test = True
    lambda_ = 1.

    times_t_dict, times_T, n_replications, iterations = running_test(test, device)
    X_tvar_1, X_tvar_1_replications, X_dict = simulation_L_rep_process(d, times_t_dict, times_T, n_replications, device, output_dir, iterations)

    ### Weights calculation for NW Conditional CDFs
    input_dir = "simulation_results"
    output_dir = "gaussian_weights_output"
    os.makedirs(output_dir, exist_ok=True)

    process = 'GaussiantvAR(1)'
    time_kernel = "uniform"
    space_kernel = "gaussian"
    gaussian_weights = computation_weights(d, lambda_, times_t_dict, times_T, iterations, n_replications, X_dict, 
                                       time_kernel, space_kernel, input_dir, output_dir, device)
    
    ### Empirical CDF calculation
    input_dir = "simulation_results"
    empirical_cdfs_iterations = empirical_cdf(times_t_dict, times_T, device, iterations, input_dir, X_tvar_1)

    ### Wasserstein distances between Average NW Conditional CDF and Empirical CDF
    input_dir = "simulation_results"
    input_dir_weights = "gaussian_weights_output"
    output_dir = "tvAR1_Wass"
    os.makedirs(output_dir, exist_ok=True)

    wass_distances_empirical_meanNW_iterations = wasserstein_distances(times_t_dict, times_T, n_replications, iterations, gaussian_weights, 
                                                                       empirical_cdfs_iterations, X_tvar_1_replications, input_dir, input_dir_weights, 
                                                                       output_dir, device, pplot=None)
    
    ### Mean Wasserstein distance
    input_dir = "tvAR1_Wass"
    output_dir = "tvAR1WassStats"
    os.makedirs(output_dir, exist_ok=True)
    wass_distances_stats = wass_stats(input_dir, output_dir, times_T, times_t_dict, iterations, wass_distances_empirical_meanNW_iterations)

    ### Plot mstar and sample time plot for T=1000
    T_samples = 1000
    plot_m_star(process, T_samples)
    plot_1_rep_process(process, d, T_samples, device)

     ### Plot of Wasserstein distances at different t/T
    input_dir = "tvAR1WassStats"
    plot_results(times_t_dict, times_T, n_replications, iterations, process, wass_distances_stats, time_kernel, space_kernel, input_dir)




if __name__ == '__main__':
    main()