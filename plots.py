#script for some simple plots, including correlation and bias of 2 variables

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import config as cfg
from corr_2d_ttest import corr_2d_ttest
from collections import namedtuple


#simple function to plot correlations between two variables
def correlation_plot(var_1, var_2, del_t, name_1, name_2):
    
    #n = var_1.shape
    #corr = np.zeros((n[1], n[2]))
    #
    #calculate correlation between both variables
    #for j in range(n[1]):
    #    print(j)
    #    for k in range(n[2]):
    #        
    #        #calculate running mean, if necessary
    #        if del_t != 1:
    #            var1_mean = np.zeros(n[0] - (del_t - 1))
    #            var2_mean = np.zeros(n[0] - (del_t - 1))
    #            for i in range(len(var1_mean)):
    #                var1_mean[i] = np.mean(var_1[i:i+del_t, j, k])
    #                var2_mean[i] = np.mean(var_2[i:i+del_t, j, k])
    #
    #        else:
    #            var1_mean = var_1[:, j, k]
    #            var2_mean = var_2[:, j, k]
    #        
    #        corr[j, k] = pearsonr(var1_mean, var2_mean)[0] 

    SET = namedtuple("SET", "nsim method alpha")
    corr, significance = corr_2d_ttest(var_1, var_2, options = SET(nsim=1000, method='ttest', alpha=0.05), nd=3)
    sig = np.where(significance==True)

    plt.figure(figsize=(10, 5))
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.scatter(sig[1], sig[0], c='black', s=5, marker='.', alpha=0.4)
    plt.colorbar()
    plt.xlabel('Longitudes')
    plt.ylabel('Latitudes')
    plt.title('Correlation between {} and {}: {} year mean'.format(str(name_1), str(name_2), str(del_t)))
    plt.savefig(cfg.tmp_path + 'plots/correlation_' + name_1 + '_' + name_2 + str(del_t) + '.pdf')
    plt.show()

#simple function to plot correlations between two variables
def bias_plot(var_1, var_2, name_1, name_2):
    
    n = var_1.shape
    bias = np.zeros((n[1], n[2]))
    var_1 = np.mean(var_1, axis=0)
    var_2 = np.mean(var_2, axis=0)


    #calculate correlation between both variables
    for j in range(n[1]):
        print(j)
        for k in range(n[2]):
            bias[j, k] = var_1[j, k] - var_2[j, k]

    

    plt.figure(figsize=(8, 5))
    plt.imshow(bias, cmap='coolwarm', vmin=-2, vmax = 2)
    plt.colorbar()
    plt.xlabel('Longitudes')
    plt.ylabel('Latitudes')
    plt.title('Bias: {} - {}'.format(str(name_1), str(name_2)))
    plt.savefig(cfg.tmp_path + 'plots/bias_' + name_1 + '_' + name_2 + '.pdf')

