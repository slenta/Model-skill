#script to plot all wanted plots

from leadyear import ly_series
import config as cfg
from leadyear import calculate_leadyear
from preprocessing import get_variable
import preprocessing as pp
from decorrelation_time import decorrelation_time
from plots import correlation_plot, plot_variable_mask
from plots import bias_plot
from residuals import residual
import matplotlib
import h5py as h5

matplotlib.use('Agg')
cfg.set_args()

#plot correlation for a specific lead year
#lys = calculate_leadyear(cfg.start_year, cfg.end_year, lead_year='2 9')
#lys.plot()

#plot leadyear correlation for all lead years
#ly = ly_series(cfg.start_year, cfg.end_year)
#ly.ly_series()

#plot decorrelation time for HadISSTs
#define start and end years, threshold for decorrelation mask
start_year = 1960
end_year = 2015
threshold = 2

#normal monthly hadissts
HadIsst = get_variable(cfg.observation_path, name='HadIsst', start_year=start_year, end_year=end_year)
HadIsst = HadIsst.__getitem__()

#historical model run, annual mean
hist = get_variable(cfg.historical_path, name=cfg.hist_name, ensemble_members=cfg.ensemble_member, start_year=start_year,
    end_year=end_year, start_month='01', variable='tos', ensemble=True, mean='annual', mode='hist')
hist = hist.__getitem__()

#annual hadisst timeseries
HadIsst_annual = get_variable(path = cfg.observation_path, name = 'HadI_annual', start_year = start_year, end_year = end_year, mean='annual')
HadIsst_annual = HadIsst_annual.__getitem__()

#create residual observation timeseries
residual_dataset = residual(lead_year=1, start_year=start_year)
print(len(HadIsst), len(hist))
HadIsst_res = residual_dataset.obs_res(HadIsst_annual, hist)

#plot residual and normal annual decorrelation times
decor = decorrelation_time(HadIsst_annual, del_t=1, threshold=threshold, name='HadIsst_annual')
dc, mask = decor.__getitem__()
decor.plot()
decor = decorrelation_time(HadIsst_annual, del_t=4, threshold=threshold, name='HadIsst_annual')
dc, mask = decor.__getitem__()
decor.plot()
decor = decorrelation_time(HadIsst_annual, del_t=1, threshold=threshold, name='HadIsst_annual')
dc, mask = decor.__getitem__()
decor.plot()

decor = decorrelation_time(HadIsst_res, del_t=1, threshold=threshold, name='HadIsst_annual')
dc, mask_decor_1 = decor.__getitem__()
decor.plot()
decor = decorrelation_time(HadIsst_res, del_t=4, threshold=threshold, name='HadIsst_annual')
dc, mask_decor_4 = decor.__getitem__()
decor.plot()
decor = decorrelation_time(HadIsst_res, del_t=1, threshold=threshold, name='HadIsst_annual')
dc, mask_decor_8 = decor.__getitem__()
decor.plot()


#plot ssh bias and correlation
Aviso_ssh = get_variable(path=cfg.tmp_path + 'Aviso_Ssh_full/Aviso_Ssh_full_2000_2018.nc', name='Aviso_ssh', start_year=2000, end_year=2017, variable='var', mean='monthly')
Aviso_ssh = Aviso_ssh.__getitem__()
Assi_ssh = get_variable(path=cfg.assi_path, name='_dcppA-assim_r', ensemble_members=cfg.ensemble_member, mod_year=cfg.ssh_mod, start_year=2000, end_year=2017,  start_month='01', start_year_file=1950, end_year_file=2017, variable='zos', ensemble=True, time_edit=True, mean='monthly')
Assi_ssh = Assi_ssh.__getitem__()

mask_ssh_8 = correlation_plot(Aviso_ssh, Assi_ssh, del_t=8, name_1='Aviso_ssh', name_2='Assimilation_ssh')
mask_ssh_4 = correlation_plot(Aviso_ssh, Assi_ssh, del_t=4, name_1='Aviso_ssh', name_2='Assimilation_ssh')
mask_ssh_1 = correlation_plot(Aviso_ssh, Assi_ssh, del_t=1, name_1='Aviso_ssh', name_2='Assimilation_ssh')
bias_plot(Aviso_ssh, Assi_ssh, name_1='Aviso_ssh', name_2='Assimilation_ssh')

#plot correlation between ocean heat content and ssts
IAP_Ohc = get_variable(path = cfg.ohc_path, name='IAP_Ohc', start_year=196100, end_year=201600, variable='heatcontent', time_edit=False)
IAP_Ohc = IAP_Ohc.__getitem__()
mask_ohc_8 = correlation_plot(HadIsst, IAP_Ohc, del_t=8, name_1='Residual_HadIsst', name_2='IAP_Ohc')
mask_ohc_4 = correlation_plot(HadIsst, IAP_Ohc, del_t=4, name_1='Residual_HadIsst', name_2='IAP_Ohc')
mask_ohc_1 = correlation_plot(HadIsst, IAP_Ohc, del_t=1, name_1='Residual_HadIsst', name_2='IAP_Ohc')

final_mask_8 = mask_decor_8 * mask_ohc_8 * mask_ssh_8
final_mask_4 = mask_decor_4 * mask_ohc_4 * mask_ssh_4
final_mask_1 = mask_decor_1 * mask_ohc_1 * mask_ssh_1

f4 = h5.File(cfg.tmp_path + 'correlation/correlation' + str(cfg.start_year) + '_' + str(cfg.end_year) + '_' + str(12) + '.hdf5', 'r')
f8 = h5.File(cfg.tmp_path + 'correlation/correlation' + str(cfg.start_year) + '_' + str(cfg.end_year) + '_' + str(20) + '.hdf5', 'r')
res_hind_4 = f4.get('res_hind_corr')
res_hind_8 = f8.get('res_hind_corr')


plot_variable_mask(res_hind_4, final_mask_4, 'Residual_hindcast_leadyear_2-5')
plot_variable_mask(res_hind_8, final_mask_8, 'Residual_hindcast_leadyear_2-9')
