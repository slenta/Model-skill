#Class to read in Hindcast, Observations and Historical runs to create residual hindcast timeseries and residual observations timeseries
#Author: Simon Lentz

from pickletools import long1
from time import time
from unicodedata import name
import numpy as np
import matplotlib.pyplot as plt
from sympy import python
import xarray as xr
from scipy.stats.stats import pearsonr
import netCDF4 as nc
import cdo
cdo = cdo.Cdo()
import h5py
import config as cfg

import tempfile
tempfile.tempdir ='/mnt/lustre02/work/uo1075/u301617/tmp'



#first step to residual: find correlation between prediction p and forcing h
#cdo timcor -selyear,${STRT_YEAR}/${END_YEAR} $HIND_FILE -subc,273.15 -selyear,${STRT_YEAR}/${END_YEAR} $HIST_FILE corrHC_r${rid}_ly${ly}.nc

#compute ration of standard deviations of p and h
#cdo div -timstd -selyear,${STRT_YEAR}/${END_YEAR} $HIND_FILE -timstd -subc,273.15 -selyear,${STRT_YEAR}/${END_YEAR} $HIST_FILE ratioHC_r${rid}_ly${ly}.nc

#muliply correlation with standard deviation ratios to find scaling factor
#cdo mul corrHC_r${rid}_ly${ly}.nc ratioHC_r${rid}_ly${ly}.nc scalingHC_r${rid}_ly${ly}.nc

#subtract scaled historical ensemble mean from full hindcast signal to calculate residual predicted signal
#cdo sub -selyear,${STRT_YEAR}/${END_YEAR} $HIND_FILE -mul scalingHC_r${rid}_ly${ly}.nc -subc,273.15 -selyear,${STRT_YEAR}/${END_YEAR} $HIST_FILE $RESH_FILE

#cdo -remapbil,reference_file file_to_convert output_file


class residual(object):
    def __init__(self, scenario_path, historical_path, hindcast_path, observation_path, final_path, start_year, end_year, temp_path, model_specifics, scenario, lon1, lon2, lat1, lat2, lead_year):
        self.histpath = historical_path
        self.scenario_path = scenario_path
        self.hindpath = hindcast_path
        self.obspath = observation_path
        self.finalpath = final_path
        self.first = start_year
        self.final = end_year
        self.start = str(self.first + 1) + '-01'
        self.end = str(self.final) + '-12'
        self.temppath = temp_path
        self.model = model_specifics
        self.scenario = scenario
        self.lon1 = lon1
        self.lon2 = lon2
        self.lat1 = lat1
        self.lat2 = lat2

        if type(lead_year) == int:
            self.lead_year = lead_year
        else:
            lead_year = lead_year.split()
            lead_year1 = int(lead_year[0])
            lead_year2 = int(lead_year[1])
            self.lead_year = lead_year1 + 2*lead_year2


    def __getitem__(self):
        
        #create outputfiles for cdo
        ofile_hist = self.temppath + 'hist/historical_newgrid_' + str(self.lead_year) + '.nc'
        ofile_hind = self.temppath + 'hind/hindcast_newgrid_' + str(self.lead_year) + '.nc'
        ofile_obs = self.temppath + 'obs/observation_newgrid_' + str(self.lead_year) + '.nc'

        #load historical, hindcast and observations
        dhis = xr.open_dataset(ofile_hist, decode_times=False)
        do = xr.load_dataset(ofile_obs, decode_times=False)
        dhin = xr.load_dataset(ofile_hind, decode_times=False)

        #decode times into day-month-year shape
        time_his = dhis.time
        dhis['time'] = nc.num2date(time_his[:],time_his.units)
        time_hind = dhin.time
        dhin['time'] = nc.num2date(time_hind[:],time_hind.units)
        time_obs = do.time
        do['time'] = nc.num2date(time_obs[:],time_obs.units)

        #select wanted timeframe
        dhis = dhis.sel(time=slice(self.start, self.end))
        dhin = dhin.sel(time=slice(self.start, self.end))
        do = do.sel(time=slice(self.start, self.end))

        #select wanted spatial frame
        dhis = dhis.sel(lon = slice(self.lon1, self.lon2))
        dhis = dhis.sel(lat = slice(self.lat1, self.lat2))
        do = do.sel(lon = slice(self.lon1, self.lon2))
        do = do.sel(lat = slice(self.lat1, self.lat2))
        dhin = dhin.sel(lon = slice(self.lon1, self.lon2))
        dhin = dhin.sel(lat = slice(self.lat1, self.lat2))

        #load sst values, reverse longitude dimension
        his = dhis.tos.values[:, ::-1, :]
        hin = dhin.tos.values[:, ::-1, :]
        o = do.sst.values[:, ::-1, :]

        #get out all NaNs
        x = np.isnan(hin)
        hin[x] = 0.1
        x = np.isnan(o)
        o[x] = 0.1
        x = np.isnan(his)
        his[x] = 0.1

        return o, his, hin

    def remap(self, infile, outfile):

        #remap grids to fit each other to coarsest grids - template 1°x1° grid
        cdo.remapbil(self.temppath + 'template.nc', input=infile, output=outfile)

    def get_coords(self):
        
        dhin = xr.load_dataset(self.temppath + 'hind/hindcast_newgrid_' + str(self.lead_year) + '.nc', decode_times=False)
        
        time_hin = dhin.time
        dhin['time'] = nc.num2date(time_hin[:],time_hin.units)
        dhin = dhin.sel(time=slice(self.start, self.end))

        lon_o = dhin.lon.values
        lat_o = dhin.lat.values
        time = dhin.time.values

        return lon_o, lat_o, time

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
        
        #fourth step: subtract scaled historical ensemble mean from hindcasts to create residual hindcasts
        scaled_hist = scaling * his
        res_hind = hind - scaled_hist

        x = np.isnan(res_hind)
        res_hind[x] = 0.1

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
        
        #fourth step: subtract scaled historical ensemble mean from hindcasts to create residual hindcasts
        scaled_hist = scaling * his
        res_obs = obs - scaled_hist

        x = np.isnan(res_obs)
        res_obs[x] = 0.1

        return res_obs

    def get_paths(self, start_year, start_year_hist, end_year_hist, start_month, ensemble_member):

        yearly_specifics_hind = str(start_year) + '-r' + str(ensemble_member) + 'i1p1f1_gn_' + str(start_year) + str(start_month) + '-' + str(start_year + 10) + '12.nc'
        yearly_specifics_hist = str(ensemble_member) + 'i1p1f1_gn_' + str(start_year_hist) + '01-' + str(end_year_hist) + '12.nc'

        hin_path = self.hindpath + str(ensemble_member) + 'i1p1f1/Omon/tos/gn/v20190821/' + self.model + '_dcppA-hindcast_s'

        if start_year_hist == 2015:
            if ensemble_member <= 3:
                his_path = self.scenario_path + str(ensemble_member) + 'i1p1f1/Omon/tos/gn/v20190627/' + self.model + '_' + self.scenario + '_r'

            else:
                his_path = self.scenario_path + str(ensemble_member) + 'i1p1f1/Omon/tos/gn/v20200623/' + self.model + '_' + self.scenario + '_r'
                yearly_specifics_hist = str(ensemble_member) + 'i1p1f1_gn_' + str(start_year_hist) + '01-' + str(2039) + '12.nc'


        else:    
            his_path = self.histpath + str(ensemble_member) + 'i1p1f1/Omon/tos/gn/v20181212/' + self.model + '_historical_r'

        hind_path = hin_path + yearly_specifics_hind
        hist_path = his_path + yearly_specifics_hist
        obs_path = self.obspath

        return hind_path, hist_path, obs_path

    def ensemble_mean(self, start_year_hind, start_year_hist, end_year_hist, start_month_hind, ensemble_member):

        hind_member = []
        hist_member = []

        for k in range(len(ensemble_member)):
                hind_path, hist_path, obs_path = self.get_paths(start_year_hind, start_year_hist, end_year_hist, start_month_hind, ensemble_member[k])
                
                #create outputfiles for cdo
                ofile_hist = self.temppath + 'hist/historical_newgrid_' + str(self.lead_year) + '.nc'
                ofile_hind = self.temppath + 'hind/hindcast_newgrid_' + str(self.lead_year) + '.nc'
                ofile_obs = self.temppath + 'obs/observation_newgrid_' + str(self.lead_year) + '.nc'
                
                #remap grids to allow for correlation calculation
                self.remap(hind_path, ofile_hind)
                self.remap(hist_path, ofile_hist)
                self.remap(obs_path, ofile_obs)

                obs_indv, hind_indv, hist_indv= self.__getitem__()
                hind_member.append(hind_indv)
                hist_member.append(hist_indv)

        obs = obs_indv
        hind = np.mean(hind_member, axis=0)
        hist = np.mean(hist_member, axis=0)
  
        return(obs, hind, hist)

    def concatenate_hists(self, hist_start_years, hist_end_years, ensemble_members, start_month_hind):
        
        hist_member = []
        hist = []

        for k in range(len(ensemble_members)):
            time = []  
            for i in range(len(hist_start_years)):
                hind_path, hist_path, obs_path = self.get_paths(1960, hist_start_years[i], hist_end_years[i], start_month_hind, ensemble_members[k])
                dhis = xr.open_dataset(hist_path, decode_times=False)
                
                time_his = dhis.time
                dhis['time'] = nc.num2date(time_his[:],time_his.units)
                dhis = dhis.sel(time=slice(1850, 2035))
                
                his = dhis.tos.values[:, ::-1, :]
                if i==0:
                    hist_member = his
                else:
                    hist_member = np.concatenate((hist_member, his), axis=0)
                time.append(time_his)
            hist.append(hist_member)

        hist_final = np.mean(hist_member, axis=0)
        n = hist_final.shape

        hist_path = self.temppath + 'hist/historical_' + self.model + '.nc'
        lon, lat, time = self.get_coords()
        time = time

        ds = xr.Dataset(data_vars=dict(historical=(["time", "x", "y"], hist_final)),
        coords=dict(lon=(["lon"], lon),lat=(["lat"], lat),time=time),
        attrs=dict(description="Complete Hindcast Data" + self.model))
        ds.to_netcdf(hist_path)
   

        ofile_hist = self.temppath + 'hist/historical_newgrid_' + self.model + '.nc'
        ofile_obs = self.temppath + 'obs/observation_newgrid_' + self.model + '.nc'

        self.remap(self.obspath, ofile_obs)
        self.remap(hist_path, ofile_hist)


    def plot(self, obs, his, hind):

        self.remap()
        res_obs = self.obs_res(obs, his)
        res_hind = self.hind_res(his, hind)

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 3, 1)
        plt.title('Observations')
        plt.imshow(obs[0], vmin = -5, vmax=30)
        plt.colorbar()
        plt.subplot(2, 3, 2)
        plt.title('Residual Observations')
        plt.imshow(res_obs[0], vmin=-5, vmax=30)
        plt.colorbar()
        plt.subplot(2, 3, 3)
        plt.imshow(hind[0], vmin=-5, vmax=30)
        plt.colorbar()
        plt.title('Hindcast')
        plt.subplot(2, 3, 4)
        plt.imshow(res_hind[0], vmin=-5, vmax=30)
        plt.colorbar()
        plt.title('Residual Hindcast')
        plt.subplot(2, 3, 5)
        plt.imshow(his[0], vmin=-5, vmax=30,)
        plt.colorbar()
        plt.title('Historical')
        plt.show()



    def save_data(self, obs, his, hind):

        #load variables from other functions
        res_hind = self.hind_res(his, hind)
        res_obs = self.obs_res(obs, his)
        lead = range(1, 11)
        lon, lat, time = self.get_coords()
        
        #create xarray Dataset with all variables
        ds = xr.Dataset(data_vars=dict(res_hind=(["time", "x", "y"], res_hind), res_obs=(["time", "x", "y"], res_obs), observation=(["time", "x", "y"], obs),
        hindcast=(["time", "x", "y"], hind), historical=(["time", "x", "y"], his)),
        coords=dict(lon=(["lon"], lon),lat=(["lat"], lat),time=time,),
        attrs=dict(description="Residual Hindcast and Observations" + self.model))

        #calculate yearly means, assign lead year coordinate
        ds = ds.groupby('time.year').mean('time')
        ds = ds.assign_coords(lead =("year", lead))

        ds.to_netcdf(self.finalpath + '_' + str(self.lead_year) + '.nc')



#dataset1 = residual('1998_2008_r9', 'Example_data/hindcast_MIRO_1998_r9.nc', 'Example_data/hindcast_MIRO_1998_r9.nc', 'Example_data/HadI_obs.nc', 'Example_data/final_res.nc', 1998, 2008, 'Example_data/')
#dataset1.remapgrids()
#dataset1.plot()
#dataset1.save_data()
#dataset1.save_obs_res()
#obs, his, hind = dataset1.__getitem__()
#res_obs = dataset1.obs_res()
#res_hind = dataset1.hind_res()
#
#print(obs.shape, his.shape, hind.shape)


