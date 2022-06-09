#script for some simple plots, including correlation and bias of 2 variables

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import config as cfg

#simple function to plot correlations between two variables
def correlation_plot(var_1, var_2, del_t, name_1, name_2):
    
    n = var_1.shape
    corr = np.zeros((n[1], n[2]))

    print(var_1.shape, var_2.shape)

    #calculate correlation between both variables
    for j in range(n[1]):
        for k in range(n[2]):
            
            #calculate running mean, if necessary
            if del_t != 1:
                var1_mean = np.zeros(n[0] - (del_t - 1))
                var2_mean = np.zeros(n[0] - (del_t - 1))
                for i in range(len(var1_mean)):
                    var1_mean[i] = np.mean(var_1[i:i+del_t, j, k])
                    var2_mean[i] = np.mean(var_2[i:i+del_t, j, k])

            else:
                var1_mean = var_1[:, j, k]
                var2_mean = var_2[:, j, k]
            
            print(np.isnan(var1_mean, var2_mean, var_1, var_2))
            corr[j, k] = pearsonr(var1_mean, var2_mean)[0]

    

    plt.figure(figsize=(8, 5))
    plt.imshow(corr)
    plt.xlabel('Longitudes')
    plt.ylabel('Latitudes')
    plt.title('Correlation between {} and {}'.format(str(name_1), str(name_2)))
    plt.savefig(cfg.tmp_path + 'plots/correlation_' + name_1 + '_' + name_2 + '.pdf')

#simple function to plot correlations between two variables
def bias_plot(var_1, var_2, del_t, name_1, name_2):
    
    n = var_1.shape
    bias = np.zeros((n[1], n[2]))

    #calculate correlation between both variables
    for j in range(n[1]):
        for k in range(n[2]):
            
            #calculate running mean, if necessary
            if del_t != 1:
                var1_mean = np.zeros(n[0] - (del_t - 1))
                var2_mean = np.zeros(n[0] - (del_t - 1))
                for i in range(len(var1_mean)):
                    var1_mean[i] = np.mean(var_1[i:i+del_t])
                    var2_mean[i] = np.mean(var_2[i:i+del_t])

            else:
                var1_mean = var_1[:, j, k]
                var2_mean = var_2[:, j, k]

            bias[j, k] = var1_mean - var2_mean

    

    plt.figure(figsize=(8, 5))
    plt.imshow(bias)
    plt.xlabel('Longitudes')
    plt.ylabel('Latitudes')
    plt.title('Correlation between {} and {}'.format(str(name_1), str(name_2)))
    plt.savefig(cfg.tmp_path + 'plots/correlation_' + name_1 + '_' + name_2 + '.pdf')

