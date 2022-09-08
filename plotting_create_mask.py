#script to plot all wanted plots

from leadyear import ly_series
import config as cfg
from leadyear import calculate_leadyear
from preprocessing import get_variable
from preprocessing import detrend_linear_numpy
import preprocessing as pp
from decorrelation_time import decorrelation_time
from plots import correlation_plot, plot_variable_mask
from plots import bias_plot
from residuals import residual
import matplotlib
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr

matplotlib.use('Agg')
cfg.set_args()

if not os.path.exists(cfg.plot_path):
    os.makedirs(cfg.plot_path)
if not os.path.exists(cfg.tmp_path):
    os.makedirs(cfg.tmp_path)

IAP_Ohc = get_variable(path = cfg.ohc_path, name='IAP_Ohc', start_year=196100, end_year=201600, variable='heatcontent', mean='annual', time_edit=False)
IAP_Ohc = IAP_Ohc.__getitem__()
IAP_Ohc = detrend_linear_numpy(IAP_Ohc)

#plot correlation for a specific lead year
lys = calculate_leadyear(cfg.start_year, cfg.end_year, lead_year='2 5')
hind_corr, res_hind_corr, hist_corr, diff = lys.calculate_lead_corr()
lys.plot()

#plot leadyear correlation for all lead years
ly = ly_series(cfg.start_year, cfg.end_year)
hind_ly_ts, res_hind_ly_ts, hist_ly_ts = ly.ly_series()

#plot decorrelation time for HadISSTs
#define start and end years, threshold for decorrelation mask
start_year = 1970
end_year = 2017

#normal monthly hadissts
HadIsst = get_variable(cfg.observation_path, name='HadIsst', start_year=start_year, end_year=end_year)
HadIsst = HadIsst.__getitem__()

#save continent mask
continent_mask = np.where(HadIsst==0, np.nan, 1)[0, :, :]

#historical model run, annual mean
hist_annual = get_variable(cfg.historical_path, name=cfg.hist_name, ensemble_members=cfg.ensemble_member, start_year=start_year,
    end_year=end_year, start_month='01', variable='tos', ensemble=True, mean='annual', mode='hist')
hist_annual = hist_annual.__getitem__()

#historical model run, monthly mean
hist = get_variable(cfg.historical_path, name=cfg.hist_name, ensemble_members=cfg.ensemble_member, start_year=start_year,
    end_year=end_year, start_month='01', variable='tos', ensemble=True, mode='hist')
hist = hist.__getitem__()

#annual hadisst timeseries
HadIsst_annual = get_variable(path = cfg.observation_path, name = 'HadI_annual', start_year = start_year, end_year = end_year, mean='annual')
HadIsst_annual = HadIsst_annual.__getitem__()

#create residual observation timeseries
residual_dataset = residual(lead_year=1, start_year=start_year)
HadIsst_res_annual = residual_dataset.obs_res(HadIsst_annual, hist_annual) * continent_mask
HadIsst_res = residual_dataset.obs_res(HadIsst, hist) * continent_mask


#calculate decorrelation times of residual HadISSTs
decor = decorrelation_time(HadIsst_res_annual, del_t=1, threshold=2, name='HadIsst_annual_res')
dc, mask_decor_1_res = decor.__getitem__()
decor.plot()
decor = decorrelation_time(HadIsst_res_annual, del_t=4, threshold=4, name='HadIsst_annual_res')
decor_4_res, mask_decor_4_res = decor.__getitem__()
decor.plot()


#plot ssh bias and correlation
Aviso_ssh = get_variable(path=cfg.data_path + 'Aviso_Ssh_full/Aviso_Ssh_full_1993_2018.nc', name='Aviso_ssh', start_year=1993, end_year=2017, variable='var', mean='monthly')
Aviso_ssh = Aviso_ssh.__getitem__()
Aviso_ssh = detrend_linear_numpy(Aviso_ssh)

Assi_ssh = get_variable(path=cfg.assi_path, name='_dcppA-assim_r', start_year=1993, end_year=2017, variable='zos', time_edit=True, mean='monthly', mode='assim')
Assi_ssh = Assi_ssh.__getitem__()[:, 0, :, :]
Assi_ssh = detrend_linear_numpy(Assi_ssh)

corr_ssh_4, mask_ssh_4 = correlation_plot(Aviso_ssh, Assi_ssh, del_t=4, name_1='Aviso_ssh', name_2='Assimilation_ssh')
corr_ssh_1, mask_ssh_1 = correlation_plot(Aviso_ssh, Assi_ssh, del_t=1, name_1='Aviso_ssh', name_2='Assimilation_ssh')
bias_plot(Aviso_ssh, Assi_ssh, name_1='Aviso_ssh', name_2='Assimilation_ssh')

#plot correlation between ocean heat content and ssts
#get OHC data
IAP_Ohc = get_variable(path = cfg.ohc_path, name='IAP_Ohc', start_year=196100, end_year=201600, variable='heatcontent', mean='annual', time_edit=True)
IAP_Ohc = IAP_Ohc.__getitem__()
IAP_Ohc = detrend_linear_numpy(IAP_Ohc)

corr_ohc_4, mask_ohc_4 = correlation_plot(HadIsst_res_annual, IAP_Ohc, del_t=4, name_1='Residual_HadIsst', name_2='IAP_Ohc')
corr_ohc_1, mask_ohc_1 = correlation_plot(HadIsst_res_annual, IAP_Ohc, del_t=1, name_1='Residual_HadIsst', name_2='IAP_Ohc')

#combine masks to final mask
final_mask_4_res = mask_decor_4_res * mask_ohc_4 * mask_ssh_4
final_mask_1_res = mask_decor_1_res * mask_ohc_1 * mask_ssh_1

f4 = h5.File(cfg.tmp_path + 'correlation/correlation' + str(cfg.start_year) + '_' + str(cfg.end_year) + '_' + str(12) + '.hdf5', 'r')
f1 = h5.File(cfg.tmp_path + 'correlation/correlation' + str(cfg.start_year) + '_' + str(cfg.end_year) + '_' + str(1) + '.hdf5', 'r')

#plot final masks on residual hindcast correlation
res_hind_4 = f4.get('res_hind_corr')
res_hind_1 = f1.get('res_hind_corr')
plot_variable_mask(res_hind_1, final_mask_1_res, 'Residual_res_hindcast_leadyear_1')
plot_variable_mask(res_hind_4, final_mask_4_res, 'Residual_res_hindcast_leadyear_2-5')


#get latitude, longitudes for xarray Dataset
ds_template = xr.open_dataset(f'{cfg.data_path}template.nc')
lat = ds_template.lat.values
lon = ds_template.lon.values

#create xarray Dataset with all variables
ds = xr.Dataset(
    data_vars=dict(decorrelation_mask=(["x", "y"], mask_decor_4_res), ohc_correlation_mask=(["x", "y"], mask_ohc_4), ssh_mask=(["x", "y"], mask_ssh_4), 
    decorrelation_time=(["x", "y"], decor_4_res), ohc_correlation=(["x", "y"], corr_ohc_4), ssh_correlation=(["x", "y"], corr_ssh_4), 
    hindcast_correlation=(["x", "y"], hind_corr), historical_correlation=(["x", "y"], hist_corr), residual_hindcast_correlation=(["x", "y"], res_hind_corr), difference_hindcast_historical=(["x", "y"], diff),
    hindcast_leadyear_timeseries=(hind_ly_ts), residual_hindcast_leadyear_timeseries=(res_hind_ly_ts), historical_leadyear_timeseries=(hist_ly_ts)),
    coords=dict(lon=(["lon"], lon),lat=(["lat"], lat)),
    attrs=dict(description=f'Individual significance masks for model {cfg.model_specifics_hind}')
)

#save final dataset with all masks and images
ds.to_netcdf(f'for_vimal/final_masks_{cfg.model_specifics_hind}.nc')
