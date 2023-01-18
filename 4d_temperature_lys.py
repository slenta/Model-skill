# calculate leadyear hindcast 4d temperature fields

import numpy as np
import config as cfg
from preprocessing import get_variable
import preprocessing as prepro
import xarray as xr
import h5py


# return important parameters
print(cfg.model_specifics_hind)
print(cfg.lead_year)

start_year = cfg.start_year
end_year = cfg.end_year


timeseries = []
for year in range(start_year, end_year):
    print(year)
    hind = get_variable(
        path=cfg.hindcast_path,
        lead_year=cfg.lead_year,
        name=cfg.hind_name,
        # ensemble_members=cfg.ensemble_member,
        ensemble_members=1,
        mod_year=cfg.hind_mod,
        start_year=start_year,
        end_year=end_year + cfg.hind_length,
        start_month=cfg.start_month_hind,
        start_year_file=start_year,
        end_year_file=start_year + cfg.hind_length,
        variable="thetao",
        ensemble=True,
        mean="annual",
        remap=False,
    )
    thetao = hind.get_ly()
    print(thetao.shape)
    timeseries.append(thetao)


# get latitude, longitudes for xarray Dataset
ens = prepro.ensemble_means(
    path=cfg.hindcast_path,
    lead_year=cfg.lead_year,
    name=cfg.hind_name,
    ensemble_members=cfg.ensemble_member,
    mod_year=cfg.hind_mod,
    start_year=start_year,
    end_year=end_year + cfg.hind_length,
    start_month=cfg.start_month_hind,
    start_year_file=start_year,
    end_year_file=start_year + cfg.hind_length,
    variable="thetao",
    mean="annual",
    remap=False,
)
path = ens.get_paths(1)
# template_path = f"{cfg.data_path}template.nc"
ds_template = xr.open_dataset(path)

lat = ds_template.latitude.values
lon = ds_template.longitude.values
time = range(start_year, end_year)
depth = ds_template.lev.values


timeseries = np.array(timeseries)
dims = range(0, 3)
dnames = ["time", "lat", "lon"]

# control if shapes are correct
print(lat.shape, lon.shape, depth.shape)
print(timeseries.shape)


# convert timeseries to xarray Dataset
ds = xr.Dataset(
    data_vars=dict(timeseries=(["time", "depth", "x", "y"], timeseries)),
    coords=dict(
        time=(["time"], time),
        depth=(["depth"], depth),
        lon=(["x", "y"], lon),
        lat=(["x", "y"], lat),
    ),
    attrs=dict(
        description=f"4d temperature leadyear timeseries for model {cfg.model_specifics_hind} for leadyear {cfg.lead_year}"
    ),
)

# save final dataset
ds.to_netcdf(
    f"for_vimal/4d_temperature_{cfg.model_specifics_hind}_{cfg.region}_ly_{cfg.lead_year}.nc"
)
