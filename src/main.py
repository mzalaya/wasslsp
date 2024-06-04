# Author: Mokhtar Z. Alaya <alayaelm@utc.fr>
# License:

import warnings
warnings.filterwarnings('ignore')

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
params = {'axes.labelsize': 12, # 12
          'font.size': 12, # 12
          'legend.fontsize': 12, # 12
          'xtick.labelsize': 12, # 10
          'ytick.labelsize': 12, # 10
          # 'text.usetex': True,
          'figure.figsize': (10, 8)}
plt.rcParams.update(params)

import seaborn as sns

import numpy as np
import pandas as pd
import scipy as scp
import mlflow

from src.utils import *
from src.kernels import Kernel

from scipy.stats import wasserstein_distance


def main(tvAR_type, times_t, times_T, d, n_replications, space_kernel, time_kernel, zeta, C, path_result):
    """
    Main function for running the main function
    :param tvAR_type: (str) Type of TVAR
    :param times_t: (list),
    :param times_T: (list),
    :param d: (int) Dimensionality
    :param n_replications: (int), Number of replications
    :param space_kernel: (str), Type of space kernel
    :param time_kernel: (str), Type of time kernel
    :param zeta: (float),
    :param C: (float),
    :param path_result: (str), Path to result
    :param plot: (bool), Whether to plot the result
    :return:
        """
    experiment_name = tvAR_type
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        #tell mlflow to start logging

        mlflow.log_param("fixed_times", times_t)
        mlflow.log_param("sample_times", times_T)
        mlflow.log_param("d", d)
        mlflow.log_param("n_replications", n_replications)
        mlflow.log_param("space_kernel", space_kernel)
        mlflow.log_param("time_kernel", time_kernel)
        mlflow.log_param("zeta", zeta)
        mlflow.log_param("C", C)

        # -------------------------------------------------------------------------------------------------------------
        # 1. Construct of:
        #   - X_tvar_2_replications (dictionary) where its items are a matrices of shape (n_replications, T)
        # replications of the LSP {Y_{t,T}}
        #   -  X_dict (dictionary), its items are a matrices of shape (T, d).
        print("1. Constructiong of X_tvar_2_replications and X_dict")

        X_tvar_2_replications = {}
        X_dict = {}

        for T in times_T:
            X_tvar_2_replications[f"T:{T}"] = np.zeros((n_replications, T))
            X_dict[f"T:{T}"] = {}

        for T in times_T:
            if tvAR_type == "gaussiantvAR(2)":
                phi_star = lambda u: 1.4 + np.sin(2 * np.pi * u)
                psi_star = lambda u: 1.05
                sigma_star = lambda u: 1.0
                m_star = lambda u, x, y: 2 / (psi_star(u)) * np.cos(phi_star(u)) * x - 1 / (psi_star(u) ** 2) * y
                for replication in range(n_replications):
                    epsilon = np.random.normal(size=T)
                    X = np.zeros((T, d))
                    X_tvar_2_T = np.zeros(T)
                    t = 2
                    while t <= T - 1:
                        X_tvar_2_T[t] = m_star(t/T, X_tvar_2_T[t - 1], X_tvar_2_T[t - 2]) + sigma_star(t/T) * epsilon[t]
                        X[t] = [X_tvar_2_T[t - 1], X_tvar_2_T[t - 2]]
                        t += 1
                    X_tvar_2_replications[f"T:{T}"][replication] = X_tvar_2_T
                    X_dict[f"T:{T}"][str(replication)] = X

            elif tvAR_type == "cauchytvAR(2)":
                phi_star = lambda u: 1.8 * np.cos(1.5 - np.cos(2 * np.pi * u))
                psi_star = lambda u: -0.81
                sigma_star = lambda u: 1.0
                m_star = lambda u, x, y: phi_star(u) * x + psi_star(u) * y
                for replication in range(n_replications):
                    epsilon = np.random.standard_cauchy(size=T)
                    X = np.zeros((T, d))
                    X_tvar_2_T = np.zeros(T)
                    t = 2
                    while t <= T-1:
                        X_tvar_2_T[t] = m_star(t/T, X_tvar_2_T[t-1], X_tvar_2_T[t-2]) + sigma_star(t/T) * epsilon[t]
                        X[t] = [X_tvar_2_T[t-1], X_tvar_2_T[t-2]]
                        t += 1
                    X_tvar_2_replications[f"T:{T}"][replication] = X_tvar_2_T
                    X_dict[f"T:{T}"][str(replication)] = X

            elif tvAR_type == "tvQAR(1)":
                phi_star = lambda u, U: (1.9 * U - 0.95) * u + (-1.9 * U + 0.95) * (1 - u) * U
                psi_star = lambda u: 1.0
                sigma_star = lambda u: 1.0
                m_star = lambda u, U, x: phi_star(u, U) * x
                for replication in range(n_replications):
                    t = 1
                    X_tvar_2_T = np.zeros(T)
                    epsilon = np.random.uniform(size=T)
                    X = np.zeros((T, d))
                    while t <= T - 1:
                        X_tvar_2_T[t] = m_star(t/T, epsilon[t], X_tvar_2_T[t-1]) + sigma_star(t/T) * epsilon[t] - 0.5
                        X[t] = [X_tvar_2_T[t-1]]
                        t += 1

                    X_tvar_2_replications[f"T:{T}"][replication] = X_tvar_2_T
                    X_dict[f"T:{T}"][str(replication)] = X

        # -------------------------------------------------------------------------------------------------------------
        # 2. X_tvar_2 (dictionary), for a fixed t its contains n_replications of the process.
        print("2. Constructiong of X_tvar_2")
        X_tvar_2 = {}

        for t in times_t:
            for T in times_T:
                X_tvar_2[f"t:{t}_T:{T}"] = {}

        for t in times_t:
            for T in times_T:
                X_tvar_2[f"t:{t}_T:{T}"] = []

        for t in times_t:
            for replication in range(n_replications):
                for T in times_T:
                    z = X_tvar_2_replications[f"T:{T}"][replication][t-1]
                    X_tvar_2[f"t:{t}_T:{T}"].append(z)

        for t in times_t:
            for T in times_T:
                X_tvar_2[f"t:{t}_T:{T}"] = np.array(X_tvar_2[f"t:{t}_T:{T}"])

        norm_X_tvar_2 = {}
        for t in times_t:
            for T in times_T:
                norm_X_tvar_2[f"t:{t}_T:{T}"] = scp.stats.norm.cdf(X_tvar_2[f"t:{t}_T:{T}"])

        # -------------------------------------------------------------------------------------------------------------
        # 3. NW estimator for the conditional mean function
        print("3. Nadarawa Watson estimator")

        gaussian_kernel = {}
        gaussian_weights = {}
        xi = zeta / (d + 1)
        for t in times_t:
            for T in times_T:
                gaussian_weights[f"t:{t}_T:{T}"] = {}

        for T in times_T:
            bandwidth = T ** (-xi) / C
            gaussian_kernel[f"T:{T}"] = Kernel(T=T, bandwidth=bandwidth, space_kernel=space_kernel, time_kernel=time_kernel)

        # -------------------------------------------------------------------------------------------------------------
        # 4. Fit kernels
        print("4. Fit kernels")
        for replication in range(n_replications):
            for t in times_t:
                for T in times_T:
                    gaussian_weights[f"t:{t}_T:{T}"][str(replication)] = \
                        gaussian_kernel[f"T:{T}"].fit(X_dict[f"T:{T}"][str(replication)], t-1)

        # -------------------------------------------------------------------------------------------------------------
        # 5. Get weights
        print("5. Get weights")
        gaussian_weights_tensor = {}

        for i_t in range(len(times_t)):
            for i_T in range(len(times_T)):
                gaussian_weights_tensor[f"t:{times_t[i_t]}_T:{times_T[i_T]}"] = {}

        for i_t in range(len(times_t)):
            for i_T in range(len(times_T)):
                for replication in range(n_replications):
                    gaussian_weights_tensor[f"t:{times_t[i_t]}_T:{times_T[i_T]}"][str(replication)] = \
                        gaussian_weights[f"t:{times_t[i_t]}_T:{times_T[i_T]}"][str(replication)]

        # -------------------------------------------------------------------------------------------------------------
        # 6. Empirical CDFs
        print("6. Calcul empirical CDFs")
        empirical_cds = {}
        for t in times_t:
            for T in times_T:
                empirical_cds[f"t:{t}_T:{T}"] = scp.stats.norm.cdf(X_tvar_2[f"t:{t}_T:{T}"])

        # -------------------------------------------------------------------------------------------------------------
        # 7. Get the wassersteins distances
        print("7. Get wasserstein distance")
        x_rep = {}
        y_rep = {}

        wasserstein_distances = {}

        for t in times_t:
            for T in times_T:
                x_rep[f"t:{t}_T:{str(T)}"] = np.zeros((n_replications, T))
                y_rep[f"t:{t}_T:{str(T)}"] = np.zeros((n_replications, T))
                wasserstein_distances[f"t:{t}_T:{T}"] = {}

        for replication in range(n_replications):
            for t in times_t:
                for T in times_T:
                    y, x = eval_univariate(
                        X_tvar_2_replications[f"T:{T}"][replication],
                        gaussian_weights_tensor[f"t:{t}_T:{str(T)}"][str(replication)]
                    )
                    x_rep[f"t:{t}_T:{str(T)}"][replication] = x
                    y_rep[f"t:{t}_T:{str(T)}"][replication] = y
                    distance = wasserstein_distance(y, scp.stats.norm.cdf(X_tvar_2_replications[f"T:{T}"][replication]))
                    wasserstein_distances[f"t:{t}_T:{T}"][str(replication)] = distance

        #-----------------------------------------------------------------------------------------------------------
        # 8. Wasserstein distance
        wass_distances_all_replications = {}

        for t in times_t:
            for T in times_T:
                wass_distances_all_replications[f"t:{t}_T:{T}"] = []

        for t in times_t:
            for T in times_T:
                for replications in range(n_replications):
                    wass_distances_all_replications[f"t:{t}_T:{T}"].append(
                        wasserstein_distances[f"t:{t}_T:{T}"][str(replication)])

        wass_distances_empirical_meanNW = {}
        for t in times_t:
            for T in times_T:
                wass_distances_empirical_meanNW[f"t:{t}_T:{T}"] = wasserstein_distance(
                    empirical_cds[f"t:{t}_T:{T}"], y_rep[f"t:{t}_T:{T}"].mean(axis=0),
                )

        # ---
        # 9. Save wasserstein distance in a CSV
        wass_times_t = {}
        for t in times_t:
            wass_times_t[f"t:{t}"] = []

        for t in times_t:
            for T in times_T:
                wass_times_t[f"t:{t}"].append(wass_distances_empirical_meanNW[f"t:{t}_T:{T}"])

        # log wass distance
        client = mlflow.tracking.MlflowClient()
        run_id = run.info.run_id
        wass_df = pd.DataFrame(index=times_t, columns=times_T)
        for index in wass_df.index:
            wass_df.loc[index] = np.array(wass_times_t[f"t:{index}"])

        # mlflow.log_metric(f"WassDist_for_t_{t}", wass_times_t[f"t:{t}"], step=t)
        # client.log_batch(run_id=run_id, metrics=wass_df.to_dict(orient='dict'))
        # mlflow.log_artifact("wassdistance", wass_df.to_csv())

        filename = path_result + f"Type{tvAR_type}_wass_distance{times_t}{times_T}_n_repl{n_replications}_spkernel_{space_kernel}_timekernel_{time_kernel}_zeta{zeta}_C{C}.csv"
        import csv
        with open(filename, "w") as csv_file:
            writer = csv.writer(csv_file)
            for key, value in wass_times_t.items():
                writer.writerow([key, value])
        mlflow.log_artifact(filename)

        # with open('dict.csv') as csv_file: # To read it back:
            # reader = csv.reader(csv_file)
            # mydict = dict(reader)

        print("8. Some plots")
        fig = plt.figure(figsize=(8,4))
        # plt.rcParams["figure.figsize"] = (8, 4)
        colors = plt.cm.Set1(np.linspace(0, 0., len(times_t)))
        tab20_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
            '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7',
            '#dbdb8d', '#9edae5'
        ]

        markers = ['o', '>', 'D', 'X', 'p', '*', 'H', 'v']

        for i, t in zip(range(len(times_t)), times_t):
            # plt.grid(True)
            # plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
            plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
            plt.minorticks_on()
            plt.plot(times_T, wass_times_t[f"t:{t}"], label=f"t:{t}",
                     color=tab20_colors[i], marker=markers[i], markersize=12,
                     lw=3)
            plt.xlim(np.array(times_T).min(), np.array(times_T).max())
            # plt.title(r'Wasserstein distance $W_1\big(\hat{\pi}_t(\cdot|{x}), \pi_t^\star(\cdot|{x})\big)$', fontsize=14)
            plt.xlabel(r'Sample size T', fontsize=14) #${T}$ ', fontsize=14)
            plt.ylabel(r'1D-Wasserstein distance', fontsize=14)# $W_1(\hat{\pi}_t(\cdot|{x})$',
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(loc="best")
            plt.tight_layout() # path_result +
            figname = f"Type{tvAR_type}_wass_distance_{times_t}{times_T}_n_repl{n_replications}_spkernel_{space_kernel}_timekernel_{time_kernel}_zeta{zeta}_C{C}.png"
            # plt.savefig(figname, dpi=150)
        mlflow.log_figure(fig, figname)

if __name__ == "__main__":

    tvAR_type = "cauchytvAR(2)" #"gaussiantvAR(2)" # "gaussiantvAR(2)"  # gaussiantvAR(2) tvQAR(1)
    times_t = [150, 200, 250, 300, 350, 400, 450, 500]
    times_T = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    d = 2
    n_replications = 2000
    space_kernel = "silverman" # "triangle" "gaussian" # "triangle"
    time_kernel = "tricube" #"epanechnikov"
    zeta = 0.4

    C = 20
    # path_result = "../results/"
    path_result = "/Users/mzalaya/Library/CloudStorage/Dropbox/research/git/wasslsp/results/"
    main(tvAR_type, times_t, times_T, d, n_replications, space_kernel, time_kernel, zeta, C, path_result)
    print("Done!")


# http://127.0.0.1:5000