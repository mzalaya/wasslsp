import warnings
warnings.filterwarnings('ignore')

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
params = {'axes.labelsize': 8, # 12
          'font.size': 8, # 12
          'legend.fontsize': 8, # 12
          'xtick.labelsize': 8, # 10
          'ytick.labelsize': 8, # 10
          'text.usetex': True,
          'figure.figsize': (10, 8)}
plt.rcParams.update(params)

import seaborn as sns

import numpy as np
import scipy as scp

from src.utils import *
from src.kernels import Kernel

from scipy.stats import wasserstein_distance


def main(times_t, times_T, d, n_replications, space_kernel, time_kernel, zeta, C, path_result):
    """
    :param Y_t_T: (float)
    :param d: (int), dimension
    :param times_t: (list) of fixed times
    :param times_T: (list), of sample size
    :param n_replications: (int), number of replication
    :param space_kernel: (str), space kernel
    :param time_kernel: (str);
    :return:
    """

    phi_star = lambda u: 1.4 + np.sin(2 * np.pi * u)
    psi_star = lambda u: 1.05
    sigma_star = lambda u: 1.0
    m_star = lambda u, x, y: 2 / (psi_star(u)) * np.cos(phi_star(u)) * x - 1 / (psi_star(u) ** 2) * y

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
        for replication in range(n_replications):
            t = 2
            epsilon = np.random.normal(size=T)
            X = np.zeros((T, d))
            X_tvar_2_T = np.zeros(T)
            while t <= T - 1:
                X_tvar_2_T[t] = m_star(t/T, X_tvar_2_T[t - 1], X_tvar_2_T[t - 2]) + sigma_star(t/T) * epsilon[t]
                X[t] = [X_tvar_2_T[t - 1], X_tvar_2_T[t - 2]]
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
                z = X_tvar_2_replications[f"T:{T}"][replication][t - 1]
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
                    gaussian_kernel[f"T:{T}"].fit(X_dict[f"T:{T}"][str(replication)], t - 1)

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

    import csv
    with open(path_result+f"wass_distance{times_t}{times_T}_n_repl{n_replications}_spkernel_{space_kernel}_timekernel_{time_kernel}_zeta{zeta}_C{C}.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in wass_times_t.items():
            writer.writerow([key, value])

    # with open('dict.csv') as csv_file: # To read it back:
        # reader = csv.reader(csv_file)
        # mydict = dict(reader)

    print("8. Some plots")
    plt.rcParams["figure.figsize"] = (8, 4)
    colors = plt.cm.Set1(np.linspace(0, .5, len(times_t)))
    markers = ['o', '>', 'D', 'X', 'p', '*', 'H', 'v']

    for i, t in zip(range(len(times_t)), times_t):
        plt.plot(times_T, wass_times_t[f"t:{t}"], label=f"t:{t}",
                 color=colors[i], marker=markers[i], markersize=6,
                 lw=2)
        plt.xlim(np.array(times_T).min(), np.array(times_T).max())
        plt.title(r'Wasserstein distance $W_1\big(\hat{\pi}_t(\cdot|{x}), \pi_t^\star(\cdot|{x})\big)$')
        plt.xlabel(r'Sample size ${T}$ ')
        plt.ylabel(r'$W_1(\hat{\pi}_t(\cdot|{x})$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(path_result+f"wass_distance_{times_t}{times_T}_n_repl{n_replications}_spkernel_{space_kernel}_timekernel_{time_kernel}_zeta{zeta}_C{C}.pdf", dpi=150)


if __name__ == "__main__":

    times_t = [150, 200, 250, 300, 350, 400, 450, 500]
    times_T = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    d = 2
    n_replications = 1000
    space_kernel = "gaussian" # "triangle"
    time_kernel = "uniform" #"epanechnikov"
    zeta = 0.4

    C = 20
    # path_result = "../results/"
    path_result = "/Users/mzalaya/Library/CloudStorage/Dropbox/research/git/wasslsp/results/"
    main(times_t, times_T, d, n_replications, space_kernel, time_kernel, zeta, C, path_result)
    print("Done!")
