#File to calculate residual observations and hindcasts

from matplotlib.pyplot import hist
from preprocessing import ensemble_means
import numpy as np
from scipy.stats import pearsonr
import xarray as xr
import config as cfg

cfg.set_args()

class residual(object):

    def __init__(self, lead_year, start_year):

        self.lead_year = lead_year
        self.start_year = start_year

    def hind_res(self, his, hind):
        
        n = hind.shape
        acc = np.zeros((n[1], n[2]))

        #first step: calculate correlation between hindcast and historical
        for j in range(n[1]):
            for k in range(n[2]):
                acc[j, k] = pearsonr(hind[:, j, k], his[:, j, k])[0]
        
        #second step: calculate ratios of standard deviations
        std_hind = np.std(hind, axis=0)
        std_his = np.std(his, axis=0)
        ratio = std_hind/std_his

        #third step: multiply standard deviation and correlation to find scaling factor
        scaling = ratio * acc
        scaling = np.array(scaling)
        
        #fourth step: subtract scaled historical ensemble mean from hindcasts to create residual hindcasts
        scaled_hist = scaling * his
        res_hind = hind - scaled_hist

        np.nan_to_num(res_hind, copy=False, nan=0.1)

        return res_hind

    def obs_res(self, obs, his):

        n = obs.shape
        acc = np.zeros((n[1], n[2]))

        #first step: calculate correlation between hindcast and historical
        for j in range(n[1]):
            for k in range(n[2]):
                acc[j, k] = pearsonr(obs[:, j, k], his[:, j, k])[0]

        #second step: calculate ratios of standard deviations
        std_obs = np.std(obs, axis=0)
        std_his = np.std(his, axis=0)
        ratio = std_obs/std_his

        #third step: multiply standard deviation and correlation to find scaling factor
        scaling = ratio * acc
        scaling = np.array(scaling)

        #fourth step: subtract scaled historical ensemble mean from hindcasts to create residual hindcasts
        scaled_hist = scaling * his
        res_obs = obs - scaled_hist

        np.nan_to_num(res_obs, copy=False, nan=0.1)

        return res_obs


    def save_data(self, obs, his, hind, time, lon, lat):

        #load variables from other functions
        res_hind = self.hind_res(his, hind)
        res_obs = self.obs_res(obs, his)
        lead = range(1, 11)
        
        #create xarray Dataset with all variables
        ds = xr.Dataset(data_vars=dict(res_hind=(["time", "x", "y"], res_hind), res_obs=(["time", "x", "y"], res_obs), observation=(["time", "x", "y"], obs),
        hindcast=(["time", "x", "y"], hind), historical=(["time", "x", "y"], his)),
        coords=dict(lon=(["lon"], lon),lat=(["lat"], lat),time=time,),
        attrs=dict(description="Residual Hindcast and Observations" + cfg.model_specifics))

        #calculate yearly means, assign lead year coordinate
        ds = ds.groupby('time.year').mean('time')
        ds = ds.assign_coords(lead =("year", lead))

        ds.to_netcdf(cfg.residual_path + '_' + str(self.start_year) + '_' + str(self.lead_year) + '.nc')