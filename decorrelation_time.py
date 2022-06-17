from rsa import sign
from preprocessing import get_variable
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import config as cfg
from scipy.stats import pearsonr
import h5py as h5


#class to calculate decorrelation time for two datasets
class decorrelation_time(object):

    def __init__(self, variable, del_t, threshold, name):

        self.variable = variable
        self.name = name + '_' + str(del_t)
        self.del_t = del_t
        self.threshold = threshold

    def __getitem__(self):
        
        #get variable
        var = self.variable
        n = var.shape
                    
        #decorrelation time for each point in space
        decor = np.zeros((n[1], n[2]))
        significance = np.zeros((n[1], n[2]))

        plt.imshow(var[0])
        plt.show()
                
        for i in range(n[1]):
            print(i)
            for j in range(n[2]):

                #calculate running mean, if necessary
                if self.del_t != 1:
                    var_mean = np.zeros(n[0] - (self.del_t - 1))
                    for k in range(len(var_mean)):
                        var_mean[k] = np.mean(var[k:k+self.del_t, i, j], axis=0)

                else:
                    var_mean = var[:, i, j]

                print(var_mean)

                #calculate autocorrelation: autocorrelation[k] is correlation at lag k, throw out lag 0
                autocor = sm.tsa.acf(var_mean, nlags=len(var_mean))
                autocor = np.nan_to_num(autocor, nan=0)

                #calculate decorrelation time for each gridpoint
                dc_criteria = autocor[1]/np.e
                print(dc_criteria, autocor)
                decor[i, j] = np.squeeze(np.where(autocor<dc_criteria))[0]
                print(decor[i, j])
                
                #calculate durban watson significance ~ 2*(1-ac)
                significance[i, j] = 2*(1 - autocor[int(decor[i, j])])

        
        mask = np.where(significance <= self.threshold, np.nan, 1)
        
        f = h5.File(cfg.tmp_path + 'decorrelation/decorrelation_time_' + self.name + '.hdf5', 'w')
        f.create_dataset('decorrelation_time', decor.shape, dtype = 'float32',data = decor)
        f.create_dataset('decor_mask', mask.shape, dtype = 'float32',data = mask)
        f.close()

        return decor, mask

    def plot(self):

        f = h5.File(cfg.tmp_path + 'decorrelation/decorrelation_time_' + self.name + '.hdf5', 'r')
        decor = f.get('decorrelation_time')
        decor = np.array(decor)
        significance = f.get('decor_mask')
        significance = np.array(significance)

        plt.figure(figsize=(8, 5))
        plt.scatter(significance, c='black', size=40)
        plt.imshow(decor, cmap='coolwarm', vmin=0, vmax=15)
        plt.xlabel('Longitudes')
        plt.ylabel('Latitudes')
        plt.title('Decorrelation time for ' + self.name)
        plt.colorbar(label='decorrelation time in years')
        plt.savefig(cfg.tmp_path + 'plots/decorrelation_time' + self.name + '.pdf')
        plt.show()
