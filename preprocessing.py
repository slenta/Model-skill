#File to concatenate climate data and calculate ensemble means

from importlib.resources import path
from statistics import mean
from tracemalloc import start
from typing_extensions import Self
from anyio import open_file
import matplotlib
import netCDF4 as nc
import xarray as xr
import numpy as np
import config as cfg
import cdo
cdo = cdo.Cdo()
import os
import matplotlib.pyplot as plt
cfg.set_args()

#class to concatenate files stored in different directories and save 

class concat_hist(object):

    def __init__(self, start_years, end_years, ensemble_members, scenario_path, scenario):
        super(concat_hist, self).__init__()

        self.start_years = start_years
        self.ensemble_members = ensemble_members
        self.scenario_path = scenario_path
        self.scenario = scenario
        self.end_years = end_years

    def get_path(self, start_year, end_year, ensemble_member):

        #specifically designed for this case of historical simulations and ssp scenario simulations
        
        yearly_specifics_hist = str(ensemble_member) + 'i1p1f1_gn_' + str(start_year) + '01-' + str(end_year) + '12.nc'

        if start_year == 2015:
            if ensemble_member <= 3:
                his_path = self.scenario_path + str(ensemble_member) + 'i1p1f1/Omon/tos/gn/v20190627/tos' + cfg.model_specifics + '_' + self.scenario + '_r'

            else:
                his_path = self.scenario_path + str(ensemble_member) + 'i1p1f1/Omon/tos/gn/v20200623/tos' + cfg.model_specifics + '_' + self.scenario + '_r'
                yearly_specifics_hist = str(ensemble_member) + 'i1p1f1_gn_' + str(start_year) + '01-' + str(2039) + '12.nc'

        else:    
            his_path = cfg.historical_path + str(ensemble_member) + 'i1p1f1/Omon/tos/gn/v20181212/tos' + cfg.model_specifics + '_historical_r'

        path = his_path + yearly_specifics_hist

        return path

    def concat(self):

        #concatenate the different variables and save in a new file

        for k in range(len(self.ensemble_members)): #mean over ensemble members

            print(k)
            hist_path = []
            for i in range(len(self.start_years)): #run over all start years and concatenate contained variables
                hist_path.append(self.get_path(self.start_years[i], self.end_years[i], self.ensemble_members[k]))
            
                ofile=cfg.tmp_path + 'hist_'
                cdo.remapbil(cfg.tmp_path + 'template.nc', input=hist_path[i], output=ofile + str(i) + '.nc')
                hist_path[i] = ofile + str(i) + '.nc'
            
            dhis = xr.merge([xr.load_dataset(hist_path[i], decode_times=False) for i in range(len(hist_path))])


            time_his = dhis.time
            dhis['time'] = nc.num2date(time_his[:],time_his.units)
            dhis = dhis.sel(time=slice('1850-01', '2035-01'))
            
            hist = dhis.tos.values[:, ::-1, :]
            time = dhis.time.values
            hist = np.array(hist)

            #get lon, lat values from template
            ds = xr.open_dataset(cfg.tmp_path + 'template.nc', decode_times=False)
            lon = ds.lon.values
            lat = ds.lat.values

            np.nan_to_num(hist, copy=False, nan=0.1)


            ds = xr.Dataset(data_vars=dict(tos=(["time", "lat", "lon"], hist)),
            coords=dict(lon=(["lon"], lon),lat=(["lat"], lat),time=time),
            attrs=dict(description="Complete Historical Data " + cfg.model_specifics))

            #os.remove(cfg.tmp_path + 'hist/historical_' + cfg.model_specifics + '_' + str(k) + '.nc')
            ds.to_netcdf(cfg.tmp_path + 'hist/historical_' + cfg.model_specifics + '_' + str(k) + '.nc')


class concat(object):

    def __init__(self, path, name, variable, start, end):
        super(concat, self).__init__()

        self.start = start
        self.variable = variable
        self.path = path
        self.name = name
        self.end = end

    def get_paths(self):
        
        paths = []

        for root, dirs, files in os.walk(self.path):
            for file in files:
                paths.append(os.path.join(root,file))

        return paths

    def concat(self):

        #concatenate the different variables and save in a new file
        paths = self.get_paths()

        print(self.path)

        for i in range(len(paths)):

            #remap to common grid
            ofile=cfg.tmp_path + 'tmp/' + self.name + str(i) + '.nc'
            cdo.remapbil(cfg.tmp_path + 'template.nc', input=paths[i], output=ofile)  
            paths[i] = ofile


        ds = xr.merge([xr.load_dataset(paths[i], decode_times=False) for i in range(len(paths))])
        
        time = ds.time
        ds['time'] = nc.num2date(time[:],time.units)
        ds = ds.sel(time=slice(self.start, self.end))

        var = np.array(ds[self.variable])
        time = ds.time

        #get lon, lat values from template
        ds = xr.open_dataset(cfg.tmp_path + 'template.nc', decode_times=False)
        lon = ds.lon.values
        lat = ds.lat.values

        np.nan_to_num(var, copy=False, nan=0.1)


        ds = xr.Dataset(data_vars=dict(var=(["time", "lat", "lon"], var)),
        coords=dict(lon=(["lon"], lon),lat=(["lat"], lat),time=time),
        attrs=dict(description="Concatenated data " + self.name + '_' + self.start + '_' + self.end))

        #os.remove(cfg.tmp_path + 'hist/historical_' + cfg.model_specifics + '_' + str(k) + '.nc')
        ds.to_netcdf(cfg.tmp_path + str(self.name) + '/' + self.name + '_' + self.start + '.nc')




#class to calculated ensemble means for a certain variable

class ensemble_means(object):

    def __init__(self, path, name, ensemble_members, mod_year, start_year, end_year, start_month, start_year_file, end_year_file, variable, lead_year, mean, mode=None):
        super(ensemble_means, self).__init__()

        self.path = path
        self.ensemble_members = ensemble_members
        self.name = name
        self.mod_year = mod_year
        self.start_year_file = start_year_file
        self.start_month = start_month
        self.end_year_file = end_year_file
        self.start_year = start_year
        self.end_year = end_year
        self.variable = variable
        self.mode = mode
        self.lead_year = lead_year
        self.mean = mean

    def __getitem__(self, path):


        #load saved regridded data
        ds = xr.load_dataset(path, decode_times=False)

        #decode times into day-month-year shape
        time = ds.time
        ds['time'] = nc.num2date(time[:],time.units)

        #select wanted timeframe
        ds = ds.sel(time=slice(str(self.start_year + 1) + str(self.start_month), str(self.end_year) + '-12'))

        if self.mean == 'monthly':
            ds = ds.resample(time='1M').mean()


        #select wanted spatial frame
        ds = ds.sel(lon = slice(cfg.lonlats[0], cfg.lonlats[1]))
        ds = ds.sel(lat = slice(cfg.lonlats[2], cfg.lonlats[3]))

        #load sst values, reverse longitude dimension
        var = ds[self.variable][:, ::-1, :]

        #get out all NaNs
        np.nan_to_num(var, copy=False, nan=0.1)

        return var
    
    def get_paths(self, ensemble_member):

        yearly_specifics = str(ensemble_member) + 'i1p1f1_gn_' + str(self.start_year_file) + self.start_month + '-' + str(self.end_year_file) + '12.nc'
        path = self.path + str(ensemble_member) + 'i1p1f1/Omon/' + self.variable + '/gn/' + self.mod_year + self.variable + cfg.model_specifics + self.name

        path = path + yearly_specifics

        return path

    def ensemble_mean(self):

        member = []

        for k in range(len(self.ensemble_members)):
                #get path to file
                
                if self.mode=='hist':
                    path = cfg.tmp_path + 'hist/historical_' + cfg.model_specifics + '_' + str(k) + '.nc'

                    indv = self.__getitem__(path)

                
                else:
                    ifile = self.get_paths(self.ensemble_members[k])


                    #create outputfiles for cdo
                    if not os.path.exists(cfg.tmp_path + self.name + '/'):
                        os.makedirs(cfg.tmp_path + self.name + '/')

                    path = cfg.tmp_path + self.name + '/' + self.name + str(k) + str(self.start_year) + '_' + str(self.lead_year) + '.nc'

                    #remap grids to allow for correlation calculation
                    #fit each other to coarsest grids - template 1°x1° grid

                    cdo.remapbil(cfg.tmp_path + 'template.nc', input=ifile, output=path)


                    indv = self.__getitem__(path)

                member.append(indv)

        
        mean = np.mean(member, axis=0)
        mean = np.array(mean)
  
        return mean


class get_variable(object):

    def __init__(self, path, lead_year=None, name=None, ensemble_members=None, mod_year=None, start_year=None, end_year=None, start_month=None, start_year_file=None, end_year_file=None, variable='sst', ensemble=False, time_edit=True, mean = None, mode=None):
        super(get_variable, self).__init__()

        self.path = path
        self.ensemble_members = ensemble_members
        self.name = name
        self.mod_year = mod_year
        self.start_year_file = start_year_file
        self.start_month = start_month
        self.end_year_file = end_year_file
        self.start_year = start_year
        self.end_year = end_year
        self.variable = variable
        self.ensemble = ensemble
        self.lead_year = lead_year
        self.time_edit = time_edit
        self.mode = mode
        self.mean = mean

    def __getitem__(self):

        if self.ensemble == True:
            
            print(self.mean)
            var_mean = ensemble_means(self.path, self.name, self.ensemble_members, self.mod_year, self.start_year, self.end_year, self.start_month, self.start_year_file, self.end_year_file, self.variable, self.lead_year, self.mean, self.mode)
            var = var_mean.ensemble_mean()

        else:

            ds = xr.load_dataset(self.path, decode_times=False)

            #decode times into day-month-year shape
            time = ds.time

            if self.time_edit == True:
                ds['time'] = nc.num2date(time[:],time.units)

                #select wanted timeframe
                ds = ds.sel(time=slice(str(self.start_year + 1) + '-01', str(self.end_year) + '-12'))

            else:
                ds = ds.sel(time=slice(self.start_year, self.end_year))

            if self.mean == 'monthly':
                ds = ds.resample(time='1M').mean()


            var = ds[self.variable]
            var = np.array(var)

        #get out all NaNs
        np.nan_to_num(var, copy=False, nan=0.1)
        
        return var

    def plot(self, name):

        var = self.__getitem__()

        plt.imshow(var[0])
        plt.savefig(cfg.tmp_path + name + '.pdf')
        plt.show()

    def get_coords(self):
        
        ofile = cfg.tmp_path + self.name + '/' + self.name + cfg.ensemble_member[0] + str(self.start_year) + '_' + str(self.lead_year) + '.nc'
        
        ds = xr.load_dataset(ofile, decode_times=False)
        time = ds.time
        ds['time'] = nc.num2date(time[:],time.units)
        ds = ds.sel(time=slice(str(self.start_year + 1) + '-01', str(self.end_year) + '-12'))


        lon = ds.lon.values
        lat = ds.lat.values
        time = ds.time.values

        return time, lon, lat



