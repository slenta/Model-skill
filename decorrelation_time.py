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
    

                #calculate autocorrelation: autocorrelation[k] is correlation at lag k, throw out lag 0
                autocor = sm.tsa.acf(var_mean, nlags=len(var_mean))
                autocor = np.nan_to_num(autocor, nan=0)



                #calculate decorrelation time for each gridpoint
                decor[i, j] = np.squeeze(np.where(autocor<1/np.e))[0]

        
        mask = np.where(decor <= self.threshold, np.nan, decor)
        mask = np.where(mask > self.threshold, 1, mask)
        
        f = h5.File(cfg.tmp_path + 'decorrelation_time_' + self.name + '.hdf5', 'w')
        f.create_dataset('decorrelation_time', decor.shape, dtype = 'float32',data = decor)
        f.create_dataset('decor_mask', mask.shape, dtype = 'float32',data = mask)
        f.close()

        return decor, mask

    def plot(self):

        f = h5.File(cfg.tmp_path + 'decorrelation_time_' + self.name + '.hdf5', 'r')
        decor = f.get('decorrelation_time')
        decor = np.array(decor)
        
        plt.figure(figsize=(8, 5))
        plt.imshow(decor, cmap='coolwarm')
        plt.xlabel('Longitudes')
        plt.ylabel('Latitudes')
        plt.title('Decorrelation time for ' + self.name)
        plt.colorbar()
        plt.savefig(cfg.tmp_path + 'plots/decorrelation_time' + self.name + '.pdf')
