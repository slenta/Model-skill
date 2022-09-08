#classes to calculate leadyear correlation and timeseries

import numpy as np
import xarray as xr
from scipy.stats import pearsonr
import h5py as h5
from residuals import residual
import config as cfg
from preprocessing import get_variable
from preprocessing import ensemble_means
import matplotlib.pyplot as plt

cfg.set_args()


class calculate_leadyear(object):
    
    def __init__(self, start_year, end_year, lead_year):
        
        self.start_year = start_year
        self.end_year = end_year
        self.lead_year = lead_year

    def __getitem__(self, start_year, end_year):
        print(start_year)
        
        #transform lead years into lead year
        if type(self.lead_year) == int:
            lead_year = self.lead_year

        else:
            lead_year = self.lead_year.split()
            lead_year1 = int(lead_year[0])
            lead_year2 = int(lead_year[1])
            lead_year = lead_year1 + 2*lead_year2



        #process variables to create residual dataset, if required choose scenario path instead of hist path
        obs = get_variable(cfg.observation_path, lead_year=lead_year, name='HadIobs', start_year=start_year, end_year=end_year)
        obs = obs.__getitem__()

        hist = get_variable(cfg.historical_path, lead_year=lead_year, name=cfg.hist_name, ensemble_members=cfg.ensemble_member_hist, start_year=start_year,
            end_year=end_year, start_month='01', variable=cfg.variable, ensemble=True, mode='hist')
        hist = hist.__getitem__()

        hin = get_variable(cfg.hindcast_path, lead_year=lead_year, name=cfg.hind_name, ensemble_members=cfg.ensemble_member, mod_year=cfg.hind_mod, start_year=start_year,
            end_year=end_year, start_month=cfg.start_month_hind, start_year_file=start_year, end_year_file=start_year + cfg.hind_length, variable=cfg.variable, ensemble=True)
        hind = hin.__getitem__()
        #time, lon, lat = hin.get_coords()

        if cfg.variable == 'thetao':
            hist = hist[:, 0, :, :]
            hind = hind[:, 0, :, :]   


        #residual_dataset = residual(lead_year, start_year)
        #residual_dataset.save_data(obs, hist, hind, time, lon, lat)

        #select only lead year from residuals
        #if type(self.lead_year) == int:
        #    ds = xr.open_dataset(cfg.residual_path + '_' + str(start_year) + '_' + str(lead_year) + '.nc', decode_times=False)
        #    ds = ds.sel(year=ds.year.values[self.lead_year - 1])
        #else:
        #    ds = xr.open_dataset(cfg.residual_path + '_' + str(start_year) + '_' + str(lead_year) + '.nc', decode_times=False)
        #    ds = ds.sel(year=slice(ds.year.values[lead_year1 - 1], ds.year.values[lead_year2 - 1])).mean('year')

        if type(self.lead_year) == int:
            hist = hist[lead_year - 1, :, :]
            hind = hind[lead_year - 1, :, :]
            obs = obs[lead_year - 1, :, :]
        else:
            hist = np.nanmean(hist[lead_year1 - 1: lead_year2 - 1, :, :], axis=0)            
            hind = np.nanmean(hind[lead_year1 - 1: lead_year2 - 1, :, :], axis=0)            
            obs = np.nanmean(obs[lead_year1 - 1: lead_year2 - 1, :, :], axis=0)            


        return obs, hind, hist#, res_obs, res_hind

    def calculate_lead_corr(self):

        #transform lead years into lead year
        if type(self.lead_year) == int:
            lead_year = self.lead_year

        else:
            lead_year = self.lead_year.split()
            lead_year1 = int(lead_year[0])
            lead_year2 = int(lead_year[1])
            lead_year = lead_year1 + 2*lead_year2


        obs = np.zeros(shape=(self.end_year - self.start_year, 180, 360))
        hind = np.zeros(shape=(self.end_year - self.start_year, 180, 360))
        hist = np.zeros(shape=(self.end_year - self.start_year, 180, 360))
        res_obs = np.zeros(shape=(self.end_year - self.start_year, 180, 360))
        res_hind = np.zeros(shape=(self.end_year - self.start_year, 180, 360))

        if type(self.lead_year) == int:
            for i in range(self.start_year - self.lead_year, self.end_year - self.lead_year):               
                #obs[i - self.start_year], hind[i - self.start_year], hist[i - self.start_year], res_obs[i - self.start_year], res_hind[i - self.start_year] = self.__getitem__(i, i+ cfg.hind_length)
                obs[i - self.start_year], hind[i - self.start_year], hist[i - self.start_year] = self.__getitem__(i, i+ cfg.hind_length)
        else:
            for i in range(self.start_year - lead_year2, self.end_year - lead_year2):               
                obs[i - self.start_year], hind[i - self.start_year], hist[i - self.start_year] = self.__getitem__(i, i+ cfg.hind_length)


        residual_dataset = residual(lead_year, self.start_year)
        residual_dataset.save_data(obs, hist, hind)

        f_res = h5.File(f'{cfg.residual_path}_{str(self.start_year)}_{str(lead_year)}.hdf5', 'r')
        res_hind = f_res.get('res_hind')
        res_obs = f_res.get('res_obs')
        

        n = hind.shape
        hind_corr = np.zeros((n[1], n[2]))
        hist_corr = np.zeros((n[1], n[2]))
        res_hind_corr = np.zeros((n[1], n[2]))

        #calculate correlation between hindcast and historical
        for j in range(n[1]):
            for k in range(n[2]):
                hind_corr[j, k] = pearsonr(hind[:, j, k], obs[:, j, k])[0]
                hist_corr[j, k] = pearsonr(hist[:, j, k], obs[:, j, k])[0]
                res_hind_corr[j, k] = pearsonr(res_hind[:, j, k], res_obs[:, j, k])[0]
       

        diff = hind_corr - hist_corr

        return hind_corr, res_hind_corr, hist_corr, diff

    def plot(self):
        
        hind_corr, res_hind_corr, hist_corr, diff = self.calculate_lead_corr()
        
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 2, 1)
        plt.title('Hindcast')
        plt.imshow(hind_corr, vmin = -1, vmax=1, cmap='coolwarm')
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.title('Historical')
        plt.imshow(hist_corr, vmin=-1, vmax=1, cmap='coolwarm')
        plt.colorbar()
        plt.subplot(2, 2, 3)
        plt.imshow(diff, vmin=-1, vmax=1, cmap='coolwarm')
        plt.colorbar()
        plt.title('Difference: Hindcast - Historical')
        plt.subplot(2, 2, 4)
        plt.imshow(res_hind_corr, vmin=-1, vmax=1, cmap='coolwarm')
        plt.colorbar()
        plt.title('Residual Hindcast')
        plt.savefig(cfg.plot_path + 'example_corr_' + str(self.lead_year) + '.pdf')
        plt.show()
    
    def save_lead_corr(self):

        hind_corr, res_hind_corr, hist_corr, diff = self.calculate_lead_corr()
        n = hind_corr.shape
        
        if type(self.lead_year) != int:
            lead_year = self.lead_year.split()
            lead_year1 = int(lead_year[0])
            lead_year2 = int(lead_year[1])
            lead_year = lead_year1 + 2*lead_year2
        else:
            lead_year = self.lead_year

        f = h5.File(cfg.tmp_path + 'correlation/correlation' + str(self.start_year) + '_' + str(self.end_year) + '_' + str(lead_year) + '.hdf5', 'w')
        dset1 = f.create_dataset('hind_corr', (n[0], n[1]), dtype = 'float32',data = hind_corr)
        dset2 = f.create_dataset('res_hind_corr', (n[0], n[1]), dtype = 'float32',data = res_hind_corr)
        dset3 = f.create_dataset('hist_corr', (n[0], n[1]), dtype = 'float32',data = hist_corr)
        f.close()


class ly_series(object):

    def __init__(self, start_year, end_year):
        
        self.start_year = start_year
        self.end_year = end_year

    def load_lead_corr(self, lead_year):

        f = h5.File(cfg.tmp_path + 'correlation/correlation' + str(self.start_year) + '_' + str(self.end_year) + '_' + str(lead_year) + '.hdf5', 'r')
        
        hind_corr = f.get('hind_corr')
        hist_corr = f.get('hist_corr')
        res_hind_corr = f.get('res_hind_corr')

        return hind_corr, hist_corr, res_hind_corr

    def ly_series(self):

        hind_ly_ts = []
        res_hind_ly_ts = []
        hist_ly_ts = []

        for i in range(1, cfg.hind_length):
            hind_corr, res_hind_corr, hist_corr = self.load_lead_corr(lead_year = i)
            
            hind_ly_ts.append(np.nanmean(np.nanmean(hind_corr, axis=0), axis=0))
            res_hind_ly_ts.append(np.nanmean(np.nanmean(res_hind_corr, axis=0), axis=0))
            hist_ly_ts.append(np.nanmean(np.nanmean(hist_corr, axis=0), axis=0))


        hind_corr_25, res_hind_corr_25, hist_corr_25 = self.load_lead_corr(lead_year=12)
        hind_corr_29, res_hind_corr_29, hist_corr_29 = self.load_lead_corr(lead_year=20)

        hind_ly_ts.append(np.nanmean(np.nanmean(hind_corr_25, axis=0), axis=0))
        res_hind_ly_ts.append(np.nanmean(np.nanmean(res_hind_corr_25, axis=0), axis=0))
        hist_ly_ts.append(np.nanmean(np.nanmean(hist_corr_25, axis=0), axis=0))
        
        hind_ly_ts.append(np.nanmean(np.nanmean(hind_corr_29, axis=0), axis=0))
        res_hind_ly_ts.append(np.nanmean(np.nanmean(res_hind_corr_29, axis=0), axis=0))
        hist_ly_ts.append(np.nanmean(np.nanmean(hist_corr_29, axis=0), axis=0))

        print(hind_ly_ts, res_hind_ly_ts, hist_ly_ts)
        x = range(1, cfg.hind_length + 2)

        fig, ax = plt.subplots()
        ax.plot(x, hind_ly_ts, 'x', label='Hindcast correlation')
        ax.plot(x, res_hind_ly_ts, 'x', label='Residual hindcast correlation')
        ax.plot(x, hist_ly_ts, 'x', label='Historical correlation')
        ax.set_xlabel('Leadyears')
        ax.set_ylabel('Anomaly Correlation')
        ax.set_xticks(x)
        ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '2-5', '2-9'])
        ax.grid()
        fig.suptitle('Hindcast Correlation by Lead Year: {} -- {}'.format(str(cfg.start_year), str(cfg.end_year)))
        plt.legend()
        plt.savefig('{}leadyear_timeseries_{}_{}.pdf'.format(cfg.plot_path, str(cfg.start_year), str(cfg.end_year)))
        plt.show()

        return np.array(hind_ly_ts), np.array(res_hind_ly_ts), np.array(hist_ly_ts)