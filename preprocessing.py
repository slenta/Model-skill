# File to concatenate climate data and calculate ensemble means

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
import pandas as pd
import matplotlib.pyplot as plt

cfg.set_args()


# functions to detrend xarray datasets and numpy arrays


# define a function to compute a linear trend of a timeseries
def linear_trend(x):
    pf = np.polyfit(x.time, x, 1)
    # need to return an xr.DataArray for groupby
    return xr.DataArray(pf[0])


def detrend_linear_numpy(x):
    # assumed shape of numpy array: (time, lat, lon)
    time = np.arange(x.shape[0])
    lat = np.arange(x.shape[1])
    lon = np.arange(x.shape[2])

    da = xr.DataArray(
        x, coords={"time": time, "lat": lat, "lon": lon}, dims=["time", "lat", "lon"]
    )

    # stack lat and lon into a single dimension called allpoints
    stacked = da.stack(allpoints=["lat", "lon"])
    # apply the function over allpoints to calculate the trend at each point
    trend = stacked.groupby("allpoints").apply(linear_trend)
    # unstack back to lat lon coordinates
    trend_unstacked = trend.unstack("allpoints")

    print(x.shape, trend_unstacked.to_numpy().shape)

    return x - trend_unstacked.to_numpy()


def detrend_datarray(da, dim, deg=1):

    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(dim, p.polyfit_coefficients)

    return da - fit


# class to concatenate files stored in different directories and save


class concat_hist(object):
    def __init__(
        self, start_years, end_years, ensemble_members, scenario_path, scenario
    ):
        super(concat_hist, self).__init__()

        self.start_years = start_years
        self.ensemble_members = ensemble_members
        self.scenario_path = scenario_path
        self.scenario = scenario
        self.end_years = end_years

    def get_path(self, start_year, end_year, ensemble_member):

        # specifically designed for this case of historical simulations and ssp scenario simulations

        yearly_specifics_hist = (
            str(ensemble_member)
            + "i1p1f1_gn_"
            + str(start_year)
            + "01-"
            + str(end_year)
            + "12.nc"
        )

        if start_year >= 2015:
            if ensemble_member <= 10:
                his_path = f"{self.scenario_path}{str(ensemble_member)}i1p1f1/Omon/{cfg.variable}/gn/{cfg.scenario_mod}/{cfg.variable}_Omon_{cfg.model_specifics_hist}_{self.scenario}_r"

            else:
                his_path = f"{self.scenario_path}{str(ensemble_member)}i1p1f1/Omon/{cfg.variable}/gn/v20210901/{cfg.variable}_Omon_{cfg.model_specifics_hist}_{self.scenario}_r"
                yearly_specifics_hist = (
                    f"{str(ensemble_member)}i1p1f1_gn_{str(start_year)}01-203412.nc"
                )

        else:
            if ensemble_member <= 10:
                his_path = f"{cfg.historical_path}{str(ensemble_member)}i1p1f1/Omon/{cfg.variable}/gn/{cfg.hist_mod}/{cfg.variable}_Omon_{cfg.model_specifics_hist}_historical_r"

            else:
                his_path = f"{cfg.historical_path}{str(ensemble_member)}i1p1f1/Omon/{cfg.variable}/gn/v20210901/{cfg.variable}_Omon_{cfg.model_specifics_hist}_historical_r"

        path = his_path + yearly_specifics_hist

        return path

    def concat(self):

        # concatenate the different variables and save in a new file

        for k in range(1, self.ensemble_members + 1):  # mean over ensemble members

            print(k)
            hist_path = []

            for i in range(
                len(self.start_years)
            ):  # run over all start years and concatenate contained variables
                hist_path.append(
                    self.get_path(self.start_years[i], self.end_years[i], k)
                )

                ofile = f"{cfg.tmp_path}hist/hist_{cfg.model_specifics_hind}_{str(k)}_"
                cdo.remapbil(
                    cfg.data_path + "template.nc",
                    input=hist_path[i],
                    output=ofile + str(i) + ".nc",
                )
                hist_path[i] = ofile + str(i) + ".nc"

            dhis = xr.merge(
                [
                    xr.load_dataset(hist_path[i], decode_times=False)
                    for i in range(len(hist_path))
                ]
            )

            time_his = dhis.time
            dhis["time"] = nc.num2date(time_his[:], time_his.units)
            dhis = dhis.sel(time=slice("1850-01", "2035-01"))

            hist = dhis[cfg.variable]
            time = dhis.time.values
            hist = np.array(hist)

            # get lon, lat values from template
            ds = xr.open_dataset(cfg.data_path + "template.nc", decode_times=False)
            lon = ds.lon.values
            lat = ds.lat.values

            np.nan_to_num(hist, copy=False, nan=0.1)

            ds = xr.Dataset(
                data_vars=dict(tos=(["time", "lat", "lon"], hist)),
                coords=dict(lon=(["lon"], lon), lat=(["lat"], lat), time=time),
                attrs=dict(
                    description="Complete Historical Data " + cfg.model_specifics_hist
                ),
            )

            ds.to_netcdf(
                f"{cfg.tmp_path}hist/historical_{cfg.model_specifics_hist}_{str(k)}.nc"
            )


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
                paths.append(os.path.join(root, file))

        return paths

    def concat(self):

        # concatenate the differe/pool/data/CMIP6/data/DCPP/MIROC/MIROC6/dcppA-hindcast/nt variables and save in a new file
        paths = self.get_paths()

        print(self.path)

        for i in range(len(paths)):

            # remap to common grid
            ofile = cfg.tmp_path + "tmp/" + self.name + str(i) + ".nc"
            cdo.remapbil(cfg.data_path + "template.nc", input=paths[i], output=ofile)
            paths[i] = ofile

        ds = xr.merge(
            [xr.load_dataset(paths[i], decode_times=False) for i in range(len(paths))]
        )

        time = ds.time
        ds["time"] = nc.num2date(time[:], time.units)
        ds = ds.sel(time=slice(self.start, self.end))

        var = np.array(ds[self.variable])
        time = ds.time

        # get lon, lat values from template
        ds = xr.open_dataset(cfg.data_path + "template.nc", decode_times=False)
        lon = ds.lon.values
        lat = ds.lat.values

        np.nan_to_num(var, copy=False, nan=0.1)

        ds = xr.Dataset(
            data_vars=dict(var=(["time", "lat", "lon"], var)),
            coords=dict(lon=(["lon"], lon), lat=(["lat"], lat), time=time),
            attrs=dict(
                description="Concatenated data "
                + self.name
                + "_"
                + self.start
                + "_"
                + self.end
            ),
        )

        ds.to_netcdf(
            cfg.tmp_path + str(self.name) + "/" + self.name + "_" + self.start + ".nc"
        )


# class to calculated ensemble means for a certain variable


class ensemble_means(object):
    def __init__(
        self,
        path,
        name,
        ensemble_members,
        mod_year,
        start_year,
        end_year,
        start_month,
        start_year_file,
        end_year_file,
        variable,
        lead_year,
        mean,
        mode=None,
        remap=True,
    ):
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
        self.remap = remap

    def __getitem__(self, path):

        # load saved regridded data
        ds = xr.load_dataset(path, decode_times=False)

        # decode times into day-month-year shape
        time = ds.time
        ds["time"] = nc.num2date(time[:], time.units)

        # select wanted timeframe
        ds = ds.sel(
            time=slice(str(self.start_year + 1) + "01", str(self.end_year) + "-12")
        )

        if self.mean == "monthly":
            ds = ds.resample(time="1M").mean()
        elif self.mean == "annual":
            ds = ds.resample(time="1Y").mean()

        if self.remap == True:
            # select wanted spatial frame
            ds = ds.sel(lon=slice(cfg.lonlats[0], cfg.lonlats[1]))
            ds = ds.sel(lat=slice(cfg.lonlats[2], cfg.lonlats[3]))

        # load sst values, reverse latitude dimension
        var = ds[self.variable]

        # get out all NaNs
        np.nan_to_num(var, copy=False, nan=0.1)

        return var

    def get_paths(self, ensemble_member):

        yearly_specifics = f"{self.variable}_Omon_{cfg.model_specifics_hind}{self.name}{str(self.start_year_file)}-r{str(ensemble_member)}i1p1f1_gn_{str(self.start_year_file)}{self.start_month}-{str(self.end_year_file)}12.nc"
        path = f"{self.path}{str(self.start_year_file)}-r{str(ensemble_member)}i1p1f1/Omon/{self.variable}/gn/{self.mod_year}/"

        path = path + yearly_specifics

        return path

    def ensemble_mean(self):

        member = []

        for k in range(1, self.ensemble_members + 1):
            # get path to file

            if self.mode == "hist":
                path = f"{cfg.tmp_path}hist/historical_{cfg.model_specifics_hist}_{str(k)}.nc"
                indv = self.__getitem__(path)

            else:

                if self.mode == "assim":

                    yearly_specifics = f"{self.variable}_Omon_{cfg.model_specifics_hind}{self.name}{str(k)}i1p1f1_gn_{str(self.start_year_file)}{self.start_month}-{str(self.end_year_file)}12.nc"
                    path = f"{self.path}{str(k)}i1p1f1/Omon/{self.variable}/gn/{self.mod_year}/"

                    path = path + yearly_specifics

                    ifile = path

                else:

                    ifile = self.get_paths(k)

                # create outputfiles for cdo
                if not os.path.exists(cfg.tmp_path + self.name + "/"):
                    os.makedirs(cfg.tmp_path + self.name + "/")

                path = f"{cfg.tmp_path}{self.name}/{self.name}{str(k)}{str(self.start_year)}_{str(self.lead_year)}.nc"

                # remap grids to allow for correlation calculation
                # fit each other to coarsest grids - template 1°x1° grid
                if self.remap == True:
                    cdo.remapbil(
                        cfg.data_path + "template.nc", input=ifile, output=path
                    )
                    indv = self.__getitem__(path)
                else:
                    indv = self.__getitem__(ifile)

            member.append(indv)

        mean = np.mean(member, axis=0)
        mean = np.array(mean)

        return mean


class get_variable(object):
    def __init__(
        self,
        path,
        lead_year=None,
        name=None,
        ensemble_members=None,
        mod_year=None,
        start_year=None,
        end_year=None,
        start_month=None,
        start_year_file=None,
        end_year_file=None,
        variable="sst",
        ensemble=False,
        time_edit=True,
        mean=None,
        mode=None,
        remap=True,
    ):
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
        self.remap = remap

    def __getitem__(self):

        if self.ensemble == True:
            var_mean = ensemble_means(
                self.path,
                self.name,
                self.ensemble_members,
                self.mod_year,
                self.start_year,
                self.end_year,
                self.start_month,
                self.start_year_file,
                self.end_year_file,
                self.variable,
                self.lead_year,
                self.mean,
                self.mode,
                self.remap,
            )
            var = var_mean.ensemble_mean()

        else:

            if self.lead_year:
                ofile = (
                    f"{cfg.tmp_path}tmp/{self.name}{self.start_year}{self.lead_year}.nc"
                )
            else:
                ofile = f"{cfg.data_path}tmp/{self.name}{str(self.start_year)}.nc"

            if self.remap == True:
                cdo.remapbil(
                    cfg.data_path + "template.nc", input=self.path, output=ofile
                )
                ds = xr.load_dataset(ofile, decode_times=False)
                # select wanted spatial frame
                ds = ds.sel(lon=slice(cfg.lonlats[0], cfg.lonlats[1]))
                ds = ds.sel(lat=slice(cfg.lonlats[2], cfg.lonlats[3]))

            else:
                ds = xr.load_dataset(self.path, decode_times=False)

            # decode times into day-month-year shape
            time = ds.time

            if self.time_edit == True:
                ds["time"] = nc.num2date(time[:], time.units)

                # select wanted timeframe
                ds = ds.sel(
                    time=slice(
                        str(self.start_year + 1) + "-01", str(self.end_year) + "-12"
                    )
                )

            else:
                time = []
                for t in ds.time.values:
                    time.append(pd.to_datetime(f"{t}", format="%Y%m.5"))

                ds["time"] = np.array(time)
                ds = ds.sel(
                    time=slice(
                        str(self.start_year + 1) + "-01", str(self.end_year) + "-12"
                    )
                )

            if self.mean == "monthly":
                ds = ds.resample(time="1M").mean()
            elif self.mean == "annual":
                ds = ds.resample(time="1Y").mean()

            var = ds[self.variable]
        var = np.array(var)

        # get out all NaNs
        np.nan_to_num(var, copy=False, nan=0.1)
        var = var[:, ::-1, :]

        return var

    def get_ly(self):

        var = self.__getitem__()
        var = var[self.lead_year]
        return var

    def plot(self):

        var = self.__getitem__()
        plt.figure()
        plt.imshow(var[0])
        plt.savefig(cfg.tmp_path + "plots/" + self.name + ".pdf")
        plt.show()

    def get_coords(self):

        ofile = f"{cfg.tmp_path}{self.name}/{self.name}1{str(self.start_year)}_{str(self.lead_year)}.nc"

        ds = xr.load_dataset(ofile, decode_times=False)
        time = ds.time
        ds["time"] = nc.num2date(time[:], time.units)
        ds = ds.sel(
            time=slice(str(self.start_year + 1) + "-01", str(self.end_year) + "-12")
        )

        lon = ds.lon.values
        lat = ds.lat.values
        time = ds.time.values

        return time, lon, lat
