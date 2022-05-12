#File to concatenate climate data and calculate ensemble means

from importlib.resources import path
from tracemalloc import start
from typing_extensions import Self
from anyio import open_file
import netCDF4 as nc
import xarray as xr
import numpy as np
import config as cfg
import cdo
cdo = cdo.Cdo()
import os

cfg.set_args()

#class to concatenate files stored in different directories and save 

class concat(object):

    def __init__(self, start_years, end_years, ensemble_members, scenario_path, scenario):
        super(concat, self).__init__()

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
                his_path = self.scenario_path + str(ensemble_member) + 'i1p1f1/Omon/tos/gn/v20190627/' + cfg.model_specifics + '_' + self.scenario + '_r'

            else:
                his_path = self.scenario_path + str(ensemble_member) + 'i1p1f1/Omon/tos/gn/v20200623/' + cfg.model_specifics + '_' + self.scenario + '_r'
                yearly_specifics_hist = str(ensemble_member) + 'i1p1f1_gn_' + str(start_year) + '01-' + str(2039) + '12.nc'

        else:    
            his_path = cfg.historical_path + str(ensemble_member) + 'i1p1f1/Omon/tos/gn/v20181212/' + cfg.model_specifics + '_historical_r'

        path = his_path + yearly_specifics_hist

        return path

    def concat(self):

        #concatenate the different variables and save in a new file

        for k in range(len(self.ensemble_members)): #mean over ensemble members
            time = []  
            hist_path = []
            for i in range(len(self.start_years)): #run over all start years and concatenate contained variables
                hist_path.append(self.get_path(self.start_years[i], self.end_years[i], self.ensemble_members[k]))

            dhis = xr.merge([xr.load_dataarray(hist_path[i], decode_times=False) for i in range(len(hist_path))])


            time_his = dhis.time
            dhis['time'] = nc.num2date(time_his[:],time_his.units)
            dhis = dhis.sel(time=slice('1850-01', '2035-01'))

            #time_indv = dhis.time.values
            #
            hist = dhis.tos.values[:, ::-1, :]
            #if i==0:
            #    hist = his
            #else:
            #    hist = np.concatenate((hist, his), axis=0)
            #time.append(time_indv)
            #os.remove(ofile)

            n = hist.shape
            time = dhis.time.values
            print(time)

            #get lon, lat values from template
            ds = xr.open_dataset(cfg.tmp_path + 'template.nc', decode_times=False)
            lon = ds.lon.values
            lat = ds.lat.values

            ds = xr.Dataset(data_vars=dict(historical=(["time", "x", "y"], hist)),
            coords=dict(lon=(["lon"], lon),lat=(["lat"], lat),time=time),
            attrs=dict(description="Complete Historical Data " + cfg.model_specifics))

            ds.to_netcdf(cfg.tmp_path + 'hist/historical_' + cfg.model_specifics + '_' + str(k) + '.nc')



#class to calculated ensemble means for a certain variable

class ensemble_means(object):

    def __init__(self, path, name, ensemble_members, mod_year, start_year, end_year, start_month, start_year_file, end_year_file, variable, lead_year, mode=None):
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

    def __getitem__(self):
        
        #define outputfile for cdo
        ofile = cfg.tmp_path + self.name + str(self.lead_year) + '.nc'

        #load saved regridded data
        ds = xr.load_dataset(ofile, decode_times=False)

        #decode times into day-month-year shape
        time = ds.time
        ds['time'] = nc.num2date(time[:],time.units)

        #select wanted timeframe
        ds = ds.sel(time=slice(self.start_year, self.end_year))

        #select wanted spatial frame
        ds = ds.sel(lon = slice(cfg.lon1, cfg.lon2))
        ds = ds.sel(lat = slice(cfg.lat1, cfg.lat2))

        #load sst values, reverse longitude dimension
        var = ds[self.variable][:, ::-1, :]

        #get out all NaNs
        x = np.isnan(var)
        var[x] = 0.1

        return var
    
    def remap(self, infile, outfile):

        #remap grids to fit each other to coarsest grids - template 1°x1° grid
        cdo.remapbil(cfg.tmp_path + 'template.nc', input=infile, output=outfile)
    
    def get_paths(self, ensemble_member):

        yearly_specifics = '-r' + str(ensemble_member) + 'i1p1f1_gn_' + str(self.start_year_file) + str(self.start_month) + '-' + str(self.end_year_file) + '12.nc'
        path = self.path + str(ensemble_member) + 'i1p1f1/Omon/tos/gn/' + self.mod_year + cfg.model_specifics + self.name

        path = path + yearly_specifics

        return path

    def ensemble_mean(self):

        member = []

        for k in range(len(self.ensemble_members)):
                #get path to file
                
                if self.mode=='hist':
                    path = cfg.tmp_path + 'hist/historical_' + cfg.model_specifics + '_' + str(k) + '.nc'
                
                else:
                    path = self.get_paths(self.ensemble_members[k])


                #create outputfiles for cdo
                ofile = cfg.tmp_path + self.name + str(self.lead_year) + '.nc'

                #remap grids to allow for correlation calculation
                self.remap(path, ofile)

                indv = self.__getitem__()
                member.append(indv)

        
        mean = np.mean(member, axis=0)
  
        return mean


class get_variable(object):

    def __init__(self, path, lead_year, name=None, ensemble_members=None, mod_year=None, start_year=None, end_year=None, start_month=None, start_year_file=None, end_year_file=None, variable='sst', ensemble=False, mode=None):
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
        self.mode = mode

    def __getitem__(self):

        if self.ensemble == True:

            var_mean = ensemble_means(self.path, self.name, self.ensemble_members, self.mod_year, self.start_year, self.end_year, self.start_month, self.start_year_file, self.end_year_file, self.variable, self.lead_year, self.mode)
            var = var_mean.ensemble_mean()

        else:

            ds = xr.load_dataset(self.path, decode_times=False)

            #decode times into day-month-year shape
            time = ds.time
            ds['time'] = nc.num2date(time[:],time.units)

            #select wanted timeframe
            ds = ds.sel(time=slice(self.start_year, self.end_year))

            var = ds[self.variable]
        
        return var

    def get_coords(self):

        ds = xr.load_dataset(self.path, decode_times=False)
        time = ds.time
        ds['time'] = nc.num2date(time[:],time.units)
        ds = ds.sel(time=slice(self.start_year, self.end_year))


        lon = ds.lon.values
        lat = ds.lat.values
        time = ds.time.values

        return time, lon, lat


            