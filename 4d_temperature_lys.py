# calculate leadyear hindcast 4d temperature fields

import numpy as np
import config as cfg
from preprocessing import get_variable
import preprocessing as prepro
import xarray as xr
import h5py


start_year = cfg.start_year
end_year = cfg.end_year
length = end_year - start_year
ly_series = np.zeros(shape=(cfg.hind_length - 1, length, 180, 360))

print(cfg.model_specifics_hind)
ly = cfg.lead_year
timeseries = []
print(ly)
for year in range(start_year, start_year + 1):
    print(year)
    hind = get_variable(
        path=cfg.hindcast_path,
        lead_year=ly,
        name=cfg.hind_name,
        ensemble_members=cfg.ensemble_member,
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
    lead_year=ly,
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


timeseries = np.array(timeseries)
print(timeseries.shape)
f = h5py.File(
    f"for_vimal/4d_temperature_{cfg.model_specifics_hind}_{cfg.region}_ly_{ly}.nc", "w"
)
f.create_dataset(name="ly_timseries", shape=timeseries.shape, data=timeseries)

dims = range(0, 3)
dnames = ["time", "lat", "lon"]

for dim, dname in zip(dims, dnames):
    h5py[cfg.data_types[0]].dims[dim].label = dname


f.close()

# convert timeseries to xarray Dataset
ds = xr.Dataset(
    data_vars=dict(timeseries=(["time", "x", "y"], timeseries)),
    coords=dict(
        time=(["time"], time),
        lon=(["lon"], lon),
        lat=(["lat"], lat),
    ),
    attrs=dict(
        description=f"4d temperature leadyear timeseries for model {cfg.model_specifics_hind} for leadyear {cfg.lead_year}"
    ),
)

# save final dataset
ds.to_netcdf(
    f"for_vimal/4d_temperature_{cfg.model_specifics_hind}_{cfg.region}_ly_{ly}.nc"
)
