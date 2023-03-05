# script for some simple plots, including correlation and bias of 2 variables

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import config as cfg
from corr_2d_ttest import corr_2d_ttest
from collections import namedtuple


# simple function to plot correlations between two variables
def correlation_plot(var_1, var_2, del_t, name_1, name_2):

    # calculate running mean, if necessary
    if del_t != 1:
        n = var_1.shape
        var_mean_1 = np.zeros((n[0] - (del_t - 1), n[1], n[2]))
        var_mean_2 = np.zeros((n[0] - (del_t - 1), n[1], n[2]))
        for k in range(len(var_mean_1)):
            var_mean_1[k, :, :] = np.mean(var_1[k : k + del_t, :, :], axis=0)
            var_mean_2[k, :, :] = np.mean(var_2[k : k + del_t, :, :], axis=0)

    else:
        var_mean_1 = var_1
        var_mean_2 = var_2

    SET = namedtuple("SET", "nsim method alpha")
    corr, significance = corr_2d_ttest(
        var_mean_1, var_mean_2, options=SET(nsim=1000, method="ttest", alpha=0.01), nd=3
    )
    sig = np.where(significance == True)

    plt.figure(figsize=(10, 5))
    plt.scatter(sig[1], sig[0], c="black", s=0.9, marker=".", alpha=0.2)
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xlabel("Longitudes")
    plt.ylabel("Latitudes")
    plt.title(
        "Correlation between {} and {}: {} year mean".format(
            str(name_1), str(name_2), str(del_t)
        )
    )
    plt.savefig(
        cfg.plot_path + "correlation_" + name_1 + "_" + name_2 + str(del_t) + ".pdf"
    )
    plt.show()

    return corr, significance


# simple function to plot correlations between two variables
def bias_plot(var_1, var_2, name_1, name_2):

    n = var_1.shape
    bias = np.zeros((n[1], n[2]))
    var_1 = np.mean(var_1, axis=0)
    var_2 = np.mean(var_2, axis=0)

    # calculate bias between both variables
    for j in range(n[1]):
        for k in range(n[2]):
            bias[j, k] = var_1[j, k] - var_2[j, k]

    plt.figure(figsize=(8, 5))
    plt.imshow(bias, cmap="coolwarm", vmin=-2, vmax=2)
    plt.colorbar()
    plt.xlabel("Longitudes")
    plt.ylabel("Latitudes")
    plt.title("Bias: {} - {}".format(str(name_1), str(name_2)))
    plt.savefig(cfg.plot_path + "bias_" + name_1 + "_" + name_2 + ".pdf")


def plot_variable_mask(var, mask, name):

    sig = np.where(mask == True)

    plt.figure(figsize=(10, 5))
    plt.scatter(sig[1], sig[0], c="black", s=0.9, marker=".", alpha=0.2)
    plt.imshow(var, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xlabel("Longitudes")
    plt.ylabel("Latitudes")
    plt.title(name)
    plt.savefig(cfg.plot_path + "variable_mask_" + name + ".pdf")
    plt.show()


# simple function to plot rmse for two variables
def rmse_plot(var_1, var_2, name_1, name_2):

    n = var_1.shape
    rmse = np.nanmean(np.sqrt((var_1 - var_2) ** 2), axis=0)

    plt.figure(figsize=(8, 5))
    plt.imshow(rmse, cmap="coolwarm", vmin=-2, vmax=2)
    plt.colorbar()
    plt.xlabel("Longitudes")
    plt.ylabel("Latitudes")
    plt.title("Bias: {} - {}".format(str(name_1), str(name_2)))
    plt.savefig(cfg.plot_path + "bias_" + name_1 + "_" + name_2 + ".pdf")

    return rmse
