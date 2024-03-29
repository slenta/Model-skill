from signal import siginterrupt
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
                dc_criteria = autocor[1]/np.e
                                
                if dc_criteria <= 0:
                    decor[i, j] = 0
                elif dc_criteria == np.nan:
                    decor[i, j] = 0
                elif dc_criteria == 0.0:
                    decor[i, j] = 0
                else:
                    decor[i, j] = np.squeeze(np.where(autocor<dc_criteria))[0]
                    
                    #calculate durban watson significance ~ 2*(1-ac)
                    #significance[i, j] = 2 * decor[i, j]

        mask = np.where(decor >= self.threshold, True, False)
        
        f = h5.File(f"{cfg.tmp_path}decorrelation/decorrelation_time_{self.name}.hdf5", 'w')
        f.create_dataset('decorrelation_time', decor.shape, dtype = 'float32',data = decor)
        f.create_dataset('decor_mask', mask.shape, dtype = 'float32',data = mask)
        f.close()

        return decor, mask

    def plot(self):

        f = h5.File(f"{cfg.tmp_path}decorrelation/decorrelation_time_{self.name}.hdf5", 'r')
        decor = f.get('decorrelation_time')
        decor = np.array(decor)
        significance = f.get('decor_mask')
        significance = np.array(significance)

        fig, ax = plt.subplots()
        im = ax.imshow(decor, cmap='coolwarm', vmin=0, vmax=15)
        ax.scatter(np.where(significance==True)[1], np.where(significance==True)[0], color='black', marker='.', s=0.8, alpha=0.2)
        ax.set_xlabel('Longitudes')
        ax.set_ylabel('Latitudes')
        ax.set_title('Decorrelation time for ' + self.name)
        plt.colorbar(im, label='decorrelation time in years')
        plt.savefig(cfg.plot_path + 'decorrelation_time_' + self.name + '.pdf')
        plt.show()
