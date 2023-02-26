# script to plot all wanted plots

from leadyear import ly_series
import config as cfg
from leadyear import calculate_leadyear
from preprocessing import get_variable
from preprocessing import detrend_linear_numpy
import preprocessing as pp
from decorrelation_time import decorrelation_time
from plots import correlation_plot, plot_variable_mask, rmse_plot, bias_plot
from residuals import residual
import matplotlib
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr

matplotlib.use("Agg")
cfg.set_args()

if not os.path.exists(cfg.plot_path):
    os.makedirs(cfg.plot_path)
if not os.path.exists(cfg.tmp_path):
    os.makedirs(cfg.tmp_path)
if not os.path.exists(f"{cfg.tmp_path}/tmp/"):
    os.makedirs(f"{cfg.tmp_path}/tmp/")
if not os.path.exists(cfg.data_path):
    os.makedirs(cfg.data_path)


# plot correlation for lead year 1 and 2-5
lys = calculate_leadyear(cfg.start_year, cfg.end_year, lead_year=1)
hind_corr_1, res_hind_corr_1, hist_corr_1, diff_1 = lys.calculate_lead_corr()
lys.plot()

lys = calculate_leadyear(cfg.start_year, cfg.end_year, lead_year="2 5")
hind_corr_4, res_hind_corr_4, hist_corr_4, diff_4 = lys.calculate_lead_corr()
lys.plot()

# plot leadyear correlation for all lead years
ly = ly_series(cfg.start_year, cfg.end_year)
hind_ly_ts, res_hind_ly_ts, hist_ly_ts = ly.ly_series()

# plot decorrelation time for HadISSTs
# define start and end years, threshold for decorrelation mask
start_year = 1970
end_year = 2017

# normal monthly hadissts
HadIsst = get_variable(
    cfg.observation_path, name="HadIsst", start_year=start_year, end_year=end_year
)
HadIsst = HadIsst.__getitem__()

# save continent mask
continent_mask = np.where(HadIsst == 0, np.nan, 1)[0, :, :]

# historical model run, annual mean
hist_annual = get_variable(
    cfg.historical_path,
    name=cfg.hist_name,
    ensemble_members=cfg.ensemble_member,
    start_year=start_year,
    end_year=end_year,
    start_month="01",
    variable="tos",
    ensemble=True,
    mean="annual",
    mode="hist",
)
hist_annual = hist_annual.__getitem__()

# historical model run, monthly mean
hist = get_variable(
    cfg.historical_path,
    name=cfg.hist_name,
    ensemble_members=cfg.ensemble_member,
    start_year=start_year,
    end_year=end_year,
    start_month="01",
    variable="tos",
    ensemble=True,
    mode="hist",
)
hist = hist.__getitem__()

# annual hadisst timeseries
HadIsst_annual = get_variable(
    path=cfg.observation_path,
    name="HadI_annual",
    start_year=start_year,
    end_year=end_year,
    mean="annual",
)
HadIsst_annual = HadIsst_annual.__getitem__()

# create residual observation timeseries
residual_dataset = residual(lead_year=1, start_year=start_year)
HadIsst_res_annual = (
    residual_dataset.obs_res(HadIsst_annual, hist_annual) * continent_mask
)
HadIsst_res = residual_dataset.obs_res(HadIsst, hist) * continent_mask


# calculate decorrelation times of residual HadISSTs
decor_1 = decorrelation_time(
    HadIsst_res_annual, del_t=1, threshold=2, name="HadIsst_annual_res"
)
decor_4 = decorrelation_time(
    HadIsst_res_annual, del_t=4, threshold=4, name="HadIsst_annual_res"
)


decor_1_res, mask_decor_1_res = decor_1.__getitem__()
decor_1.plot()
decor_4_res, mask_decor_4_res = decor_4.__getitem__()
decor_4.plot()


# plot ssh bias and correlation, if assimilation existent
if cfg.assi_path != None:
    assi_start = 1993
    assi_end = 2017

    Aviso_ssh = get_variable(
        path=cfg.data_path + "Aviso_Ssh_full/Aviso_Ssh_full_1993_2018.nc",
        name="Aviso_ssh",
        start_year=assi_start,
        end_year=assi_end,
        variable="var",
        mean="monthly",
    )
    Aviso_ssh = Aviso_ssh.__getitem__()
    Aviso_ssh = detrend_linear_numpy(Aviso_ssh)

    Assi_ssh = get_variable(
        path=cfg.assi_path,
        name="_dcppA-assim_r",
        start_year=assi_start,
        end_year=assi_end,
        variable="zos",
        time_edit=True,
        mean="monthly",
        mode="assim",
    )
    Assi_ssh = Assi_ssh.__getitem__()[:, 0, :, :]
    Assi_ssh = detrend_linear_numpy(Assi_ssh)

    corr_ssh_4, mask_ssh_4 = correlation_plot(
        Aviso_ssh, Assi_ssh, del_t=4, name_1="Aviso_ssh", name_2="Assimilation_ssh"
    )
    corr_ssh_1, mask_ssh_1 = correlation_plot(
        Aviso_ssh, Assi_ssh, del_t=1, name_1="Aviso_ssh", name_2="Assimilation_ssh"
    )
    bias_plot(Aviso_ssh, Assi_ssh, name_1="Aviso_ssh", name_2="Assimilation_ssh")

    n = Assi_ssh.shape
    Assi_initial = np.zeros(shape=(assi_end - assi_start + 1, n[1], n[2]))
    Aviso_initial = np.zeros(shape=(assi_end - assi_start + 1, n[1], n[2]))
    print(Assi_initial.shape, Assi_ssh.shape)
    for i in range(Assi_ssh.shape[0]):
        if i % int(cfg.start_month_hind) == 0:
            print(i)
            Assi_initial[i // int(cfg.start_month_hind), :, :] = Assi_ssh[i, :, :]
            Aviso_initial[i // int(cfg.start_month_hind), :, :] = Aviso_ssh[i, :, :]

    Assi_rmse = rmse_plot(
        Aviso_initial,
        Assi_initial,
        name_1="Aviso_ssh_initial_month",
        name_2="Assimilation_ssh_initial_month",
    )


# plot pi_control thetao and sst correlation
pi_control = get_variable(
    path=cfg.pi_path,
    name="Pi_control_thetao",
    start_year=cfg.start_year,
    end_year=cfg.end_year,
    variable="thetao",
    mean="annual",
    time_edit=True,
)
pi_control_thetao = pi_control.__getitem__()
pi_control_ohc_proxy = np.mean(pi_control_thetao, axis=1)
pi_control_sst = pi_control_thetao[:, 0, :, :]

pi_corr_1, pi_sig_1 = correlation_plot(
    pi_control_ohc_proxy,
    pi_control_sst,
    name_1="pi_ohc",
    name_2="pi_sst",
    del_t=1,
)
pi_corr_4, pi_sig_4 = correlation_plot(
    pi_control_ohc_proxy,
    pi_control_sst,
    name_1="pi_ohc",
    name_2="pi_sst",
    del_t=4,
)


# plot correlation between ocean heat content and ssts
# get OHC data
IAP_Ohc = get_variable(
    path=cfg.ohc_path,
    name="IAP_Ohc",
    start_year=1961,
    end_year=2016,
    variable="heatcontent",
    mean="annual",
    time_edit=True,
)
IAP_Ohc = IAP_Ohc.__getitem__()
IAP_Ohc = detrend_linear_numpy(IAP_Ohc)

corr_ohc_4, mask_ohc_4 = correlation_plot(
    HadIsst_res_annual, IAP_Ohc, del_t=4, name_1="Residual_HadIsst", name_2="IAP_Ohc"
)
corr_ohc_1, mask_ohc_1 = correlation_plot(
    HadIsst_res_annual, IAP_Ohc, del_t=1, name_1="Residual_HadIsst", name_2="IAP_Ohc"
)

# combine masks to final mask
final_mask_4_res = mask_decor_4_res * mask_ohc_4 * mask_ssh_4
final_mask_1_res = mask_decor_1_res * mask_ohc_1 * mask_ssh_1

f4 = h5.File(
    f"{cfg.tmp_path}correlation/correlation{str(cfg.start_year)}_{str(cfg.end_year)}_12.hdf5",
    "r",
)
f1 = h5.File(
    f"{cfg.tmp_path}correlation/correlation{str(cfg.start_year)}_{str(cfg.end_year)}_1.hdf5",
    "r",
)

# plot final masks on residual hindcast correlation
res_hind_4 = f4.get("res_hind_corr")
res_hind_1 = f1.get("res_hind_corr")
plot_variable_mask(res_hind_1, final_mask_1_res, "Residual_res_hindcast_leadyear_1")
plot_variable_mask(res_hind_4, final_mask_4_res, "Residual_res_hindcast_leadyear_2-5")


# get latitude, longitudes for xarray Dataset
ds_template = xr.open_dataset(f"{cfg.data_path}template.nc")
lat = ds_template.lat.values
lon = ds_template.lon.values

# create xarray Dataset with all variables
ds_1 = xr.Dataset(
    data_vars=dict(
        decorrelation_mask=(["x", "y"], mask_decor_1_res),
        ohc_correlation_mask=(["x", "y"], mask_ohc_1),
        ssh_mask=(["x", "y"], mask_ssh_1),
        decorrelation_time=(["x", "y"], decor_1_res),
        ohc_correlation=(["x", "y"], corr_ohc_1),
        hindcast_correlation=(["x", "y"], hind_corr_1),
        historical_correlation=(["x", "y"], hist_corr_1),
        residual_hindcast_correlation=(["x", "y"], res_hind_corr_1),
        difference_hindcast_historical=(["x", "y"], diff_1),
        pi_ohc_correlation=(["x", "y"], pi_corr_1),
        pi_ohc_sign=(["x", "y"], pi_sig_1),
    ),
    coords=dict(lon=(["lon"], lon), lat=(["lat"], lat)),
    attrs=dict(
        description=f"Individual significance masks for model {cfg.model_specifics_hind} annual means: leadyear 1"
    ),
)
ds_4 = xr.Dataset(
    data_vars=dict(
        decorrelation_mask=(["x", "y"], mask_decor_4_res),
        ohc_correlation_mask=(["x", "y"], mask_ohc_4),
        ssh_mask=(["x", "y"], mask_ssh_4),
        decorrelation_time=(["x", "y"], decor_4_res),
        ohc_correlation=(["x", "y"], corr_ohc_4),
        hindcast_correlation=(["x", "y"], hind_corr_4),
        historical_correlation=(["x", "y"], hist_corr_4),
        residual_hindcast_correlation=(["x", "y"], res_hind_corr_4),
        difference_hindcast_historical=(["x", "y"], diff_4),
        pi_ohc_correlation=(["x", "y"], pi_corr_4),
        pi_ohc_sign=(["x", "y"], pi_sig_4),
    ),
    coords=dict(lon=(["lon"], lon), lat=(["lat"], lat)),
    attrs=dict(
        description=f"Individual significance masks for model {cfg.model_specifics_hind} 4 year means: leadyears 2--5"
    ),
)

ds_ly = xr.Dataset(
    data_vars=dict(
        hindcast_leadyear_timeseries=(hind_ly_ts),
        residual_hindcast_leadyear_timeseries=(res_hind_ly_ts),
        historical_leadyear_timeseries=(hist_ly_ts),
    ),
    coords=dict(lon=(["lon"], lon), lat=(["lat"], lat)),
    attrs=dict(description=f"Leadyear timeseries for model {cfg.model_specifics_hind}"),
)

# if assimilation existent, save assimilation correlation and rmse plots
if cfg.assi_path != None:
    ds_assi = xr.Dataset(
        data_vars=dict(
            ssh_correlation_annual=(["x", "y"], corr_ssh_1),
            ssh_correlation_4_year=(["x", "y"], corr_ssh_4),
            assimilation_initial_rmse=(["x", "y"], Assi_rmse),
        ),
        coords=dict(lon=(["lon"], lon), lat=(["lat"], lat)),
        attrs=dict(
            description=f"Assimilation ssh masks for model {cfg.model_specifics_hind}"
        ),
    )

    ds_assi.to_netcdf(f"for_vimal/assimilation_{cfg.model_specifics_hind}.nc")


# save final dataset with all masks and images
ds_1.to_netcdf(f"for_vimal/final_masks_{cfg.model_specifics_hind}_1.nc")
ds_4.to_netcdf(f"for_vimal/final_masks_{cfg.model_specifics_hind}_4.nc")
ds_ly.to_netcdf(f"for_vimal/leadyear_timeseries_{cfg.model_specifics_hind}.nc")
