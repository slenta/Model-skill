from preprocessing import get_variable
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import config as cfg
from scipy.stats import pearsonr


#class to calculate decorrelation time for two datasets
class decorrelation_time(object):

    def __init__(self, variable, del_t, threshold):

        self.variable = variable
        self.name = str(variable) + str(del_t)
        self.del_t = del_t
        self.threshold = threshold

    def __getitem__(self):
        
        #get variable
        var = self.variable
        n = var.shape
                    
        #autocorrelation for each point in space
        ac = np.zeros((n[0] - (self.del_t - 1)-1, n[1], n[2]))
        decor = np.zeros((n[1], n[2]))
                
        for i in range(n[1]):
            for j in range(n[2]):

                print(i, j)

                #calculate running mean, if necessary
                if self.del_t != 1:
                    var_mean = np.zeros(n[0] - (self.del_t - 1))
                    for i in range(len(var_mean)):
                        var_mean[i] = np.mean(var[i:i+self.del_t])

                else:
                    var_mean = var[:, i, j]
                #calculate autocorrelation: autocorrelation[k] is correlation at lag k, throw out lag 0
                print(sm.tsa.acf(var_mean, nlags=len(var_mean))[1:].shape, var_mean.shape, ac.shape)
                autocor = sm.tsa.acf(var_mean)[1:]
                ac[:, i, j] = autocor

                #calculate decorrelation time for each 
                decor[i, j] = (1 + 2*np.sum(ac[:, i, j])) * self.del_t

        
        mask = np.where(decor <= self.threshold, np.nan, decor)
        mask = np.where(mask > self.threshold, 1, mask)

        return decor, mask

    def plot(self):

        decor, mask = self.__getitem__()

        plt.figure(figsize=(8, 5))
        plt.imshow(decor)
        plt.xlabel('Longitudes')
        plt.ylabel('Latitudes')
        plt.title('Decorrelation time for ' + self.name)
        plt.savefig(cfg.tmp_path + 'plots/decorrelation_time' + self.name + '.pdf')



#simple function to plot correlations between two variables
def correlation_plot(var_1, var_2, del_t, name):
    
    n = var_1.shape
    corr = np.zeros((n[1], n[2]))

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

            corr[j, k] = pearsonr(var1_mean[:, j, k], var2_mean[:, j, k])[0]

    

    plt.figure(figsize=(8, 5))
    plt.imshow(corr)
    plt.xlabel('Longitudes')
    plt.ylabel('Latitudes')
    plt.title('Correlation between {} and {}'.format(str(var_1), str(var_2)))
    plt.savefig(cfg.tmp_path + 'plots/correlation' + name + '.pdf')