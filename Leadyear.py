from pickletools import long1
from re import A
import string
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import ensemble
from Create_residual import residual
import xarray as xr
from scipy.stats.stats import pearsonr
import cdo
import h5py as h5
cdo = cdo.Cdo()
import config as cfg
matplotlib.use('Agg')




class calculate_leadyear(object):
    
    def __init__(self, historical_path, hindcast_path, observation_path, residual_path, start_year, end_year, hist_start_years, hist_end_years, start_month_hind, ensemble_member, model_specifics, tmp_path, scenario_path, scenario, lonlats):
        
        self.histpath = historical_path
        self.hindpath = hindcast_path
        self.obspath = observation_path
        self.resids = residual_path
        self.start_year = start_year
        self.end_year = end_year
        self.hist_years = hist_start_years
        self.hist_end_years = hist_end_years
        self.start_month_hind = start_month_hind
        self.ensemble_member = ensemble_member
        self.model = model_specifics
        self.temp = tmp_path
        self.scenariopath = scenario_path
        self.scenario = scenario
        self.lon1 = lonlats[0]
        self.lon2 = lonlats[1]
        self.lat1 = lonlats[2]
        self.lat2 = lonlats[3]

    def __getitem__(self, lead_year, start_year_hind, end_year_hind, start_year_hist, end_year_hist):
        
        #process variables to create residual dataset, if required choose scenario path instead of hist path
        residual_dataset = residual(self.scenariopath, self.histpath, self.hindpath, self.obspath, self.resids, start_year_hind, end_year_hind, self.temp, self.model, self.scenario, self.lon1, self.lon2, self.lat1, self.lat2, lead_year)
        
        obs, hist, hind = residual_dataset.ensemble_mean(start_year_hind, start_year_hist, end_year_hist, self.start_month_hind, self.ensemble_member)
        residual_dataset.save_data(obs, hist, hind)
        

        #load dataset and select lead year(s)
        if type(lead_year) == int:
            ds = xr.open_dataset(self.resids + '_' + str(lead_year) + '.nc', decode_times=False)
            ds = ds.sel(year=ds.year.values[lead_year - 1])

        else:
            lead_year = lead_year.split()
            lead_year1 = int(lead_year[0])
            lead_year2 = int(lead_year[1])
            ds = xr.open_dataset(self.resids + '_' + str(lead_year1 + 2*lead_year2) + '.nc', decode_times=False)
            ds = ds.sel(year=slice(ds.year.values[lead_year1 - 1], ds.year.values[lead_year2 - 1])).mean('year')
        
        obs = ds.observation.values
        hind = ds.hindcast.values
        hist = ds.historical.values
        res_obs = ds.res_obs.values
        res_hind = ds.res_hind.values

        return obs, hind, hist, res_obs, res_hind

    def calculate_lead_corr(self, lead_year):

        obs = np.zeros(shape=(self.end_year - self.start_year, 180, 360))
        hind = np.zeros(shape=(self.end_year - self.start_year, 180, 360))
        hist = np.zeros(shape=(self.end_year - self.start_year, 180, 360))
        res_obs = np.zeros(shape=(self.end_year - self.start_year, 180, 360))
        res_hind = np.zeros(shape=(self.end_year - self.start_year, 180, 360))

        for i in range(self.start_year, self.end_year):
            for j in range(len(self.hist_years)):
                if self.hist_years[j]>self.start_year:
                    start_year_hist = self.hist_years[j-1]
                    end_year_hist = self.hist_end_years[j-1]
                    break
                elif self.hist_years[len(self.hist_years) - 1]<= self.start_year:
                    start_year_hist = self.hist_years[len(self.hist_years) - 1]
                    end_year_hist = self.hist_end_years[len(self.hist_years) - 1]
                
            obs[i - self.start_year], hist[i - self.start_year], hind[i - self.start_year], res_obs[i - self.start_year], res_hind[i - self.start_year] = self.__getitem__(lead_year, i, i+10, start_year_hist, end_year_hist)
            

        n = hind.shape
        hind_corr = np.zeros((n[1], n[2]))
        hist_corr = np.zeros((n[1], n[2]))
        res_hind_corr = np.zeros((n[1], n[2]))

        #first step: calculate correlation between hindcast and historical
        for j in range(n[1]):
            for k in range(n[2]):
                hind_corr[j, k] = pearsonr(hind[:, j, k], obs[:, j, k])[0]
                hist_corr[j, k] = pearsonr(hist[:, j, k], obs[:, j, k])[0]
                res_hind_corr[j, k] = pearsonr(res_hind[:, j, k], res_obs[:, j, k])[0]
       

        diff = hind_corr - hist_corr

        return hind_corr, res_hind_corr, hist_corr, diff
    
    def save_lead_corr(self, lead_year):

        hind_corr, res_hind_corr, hist_corr, diff = self.calculate_lead_corr(lead_year)
        n = hind_corr.shape
        
        if type(lead_year) != int:
            lead_year = lead_year.split()
            lead_year1 = int(lead_year[0])
            lead_year2 = int(lead_year[1])
            lead_year = lead_year1 + 2*lead_year2

        f = h5.File(self.temp + 'correlation' + str(self.start_year) + '_' + str(self.end_year) + '_' + str(lead_year) + '.hdf5', 'w')
        dset1 = f.create_dataset('hind_corr', (n[0], n[1]), dtype = 'float32',data = hind_corr)
        dset2 = f.create_dataset('res_hind_corr', (n[0], n[1]), dtype = 'float32',data = res_hind_corr)
        dset3 = f.create_dataset('hist_corr', (n[0], n[1]), dtype = 'float32',data = hist_corr)
        f.close()

    def load_lead_corr(self, lead_year):

        f = h5.File(self.temp + 'correlation' + str(self.start_year) + '_' + str(self.end_year) + '_' + str(lead_year) + '.hdf5', 'r')
        
        hind_corr = f.get('hind_corr')
        hist_corr = f.get('hist_corr')
        res_hind_corr = f.get('res_hind_corr')

        return hind_corr, hist_corr, res_hind_corr

    def save_hist_final(self):

        residual_dataset = residual(self.scenariopath, self.histpath, self.hindpath, self.obspath, self.resids, 1960, 1961, self.temp, self.model, self.scenario, self.lon1, self.lon2, self.lat1, self.lat2, 2)
        residual_dataset.concatenate_hists(self.hist_years, self.hist_end_years, self.ensemble_member, self.start_month_hind)


    def plot(self, lead_year):
        
        hind_corr, res_hind_corr, hist_corr = self.load_lead_corr(lead_year)
        diff = np.array(hind_corr) - np.array(hist_corr)
        print(np.array(np.nanmean(np.nanmean(hist_corr))))
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
        plt.title('Difference')
        plt.subplot(2, 2, 4)
        plt.imshow(res_hind_corr, vmin=-1, vmax=1, cmap='coolwarm')
        plt.colorbar()
        plt.title('Residual Hindcast')
        plt.savefig('example_corr.pdf')
        #plt.show()

    def ly_series(self):

        hind_ly_ts = []
        res_hind_ly_ts = []
        hist_ly_ts = []

        for i in range(1, 11):
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
        x = range(1, 13)

        fig, ax = plt.subplots()
        ax.plot(x, hind_ly_ts, 'x')
        ax.plot(x, res_hind_ly_ts, '-')
        ax.plot(x, hist_ly_ts, 'x')
        ax.set_xlabel('Leadyears')
        ax.set_ylabel('Anomal Correlation')
        ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '2-5', '2-9'])
        ax.grid()
        fig.suptitle('Hindcast Correlation by Lead Year: 1960-2013')
        plt.savefig('Leadyear_timeseries_1960_2013.pdf')
    
    def ly_timeseries(self, lead_year):

        hind_mean = np.zeros((self.end_year - self.start_year, 180, 360))
        hist_mean = np.zeros((self.end_year - self.start_year, 180, 360))

        x = np.arange(self.start_year, self.end_year)
        
        for i in x:
            
            for j in range(len(self.hist_years) - 1):
                if self.hist_years[j]>self.start_year:
                    start_year_hist = self.hist_years[j-1]
                    end_year_hist = self.hist_end_years[j-1]
                    break
                elif self.hist_years[len(self.hist_years) - 1]<= self.start_year:
                    start_year_hist = self.hist_years[len(self.hist_years) - 1]
                    end_year_hist = self.hist_end_years[len(self.hist_years) - 1]
            
            obs, hind_mean[i - self.start_year], hist_mean[i - self.start_year], res_obs, res_hind = self.__getitem__(lead_year, i, i + 10, start_year_hist, end_year_hist)
            
        hind_mean = np.mean(np.mean(hind_mean, axis=1), axis=1)
        hist_mean = np.mean(np.mean(hist_mean, axis=1), axis=1)

        plt.plot(x, hind_mean, label='hindcast')
        plt.plot(x, hist_mean, label='historical')
        plt.grid()
        plt.title('Timeseries of lead_year' + str(lead_year))
        plt.legend()
        plt.savefig('Leadyear_timeseries_' + str(lead_year) + '.pdf')



cfg.set_args()

example_ly = calculate_leadyear(cfg.historical_path, cfg.hindcast_path, cfg.observation_path, cfg.residual_path, cfg.start_year, cfg.end_year, cfg.hist_start_years, cfg.hist_end_years, cfg.start_month_hind, cfg.ensemble_member, cfg.model_specifics, '../Example_data/', cfg.scenario_path, cfg.scenario, cfg.lonlats)
example_ly.save_hist_final()
#if cfg.lead_years:
#    example_ly.save_lead_corr(cfg.lead_years)
#else:
#    example_ly.save_lead_corr(cfg.lead_year)
#example_ly.ly_series()
#example_ly.plot(1)