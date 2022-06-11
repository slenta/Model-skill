#script to plot all wanted plots

from leadyear import ly_series
import config as cfg
from leadyear import calculate_leadyear
from preprocessing import get_variable
from decorrelation_time import decorrelation_time
from plots import correlation_plot
from plots import bias_plot
import matplotlib
from preprocessing import concat

matplotlib.use('Agg')
cfg.set_args()

#plot correlation for a specific lead year
#lys = calculate_leadyear(cfg.start_year, cfg.end_year, lead_year='2 9')
#lys.plot()

#plot leadyear correlation for all lead years
#ly = ly_series(cfg.start_year, cfg.end_year)
#ly.ly_series()

#plot decorrelation time for HadISSTs
#define threshold for decorrelation mask
threshold = 1
HadIsst = get_variable(path = cfg.observation_path, start_year = 1960, end_year = 2015)
HadIsst = HadIsst.__getitem__()
decor = decorrelation_time(HadIsst, del_t=8, threshold=threshold, name='HadIsst')
#dc, mask = decor.__getitem__()
decor.plot()


#plot ssh bias and correlation
Aviso_ssh = get_variable(path=cfg.tmp_path + 'Aviso_Ssh_full/Aviso_Ssh_full_2000-01_2019-12.nc', start_year=2000, end_year=2017, variable='var')
Aviso_ssh = Aviso_ssh.__getitem__()
Assi_ssh = get_variable(path=cfg.assi_path, ensemble_members=cfg.ensemble_member,start_year=2000, end_year=2017, start_month='01', start_year_file=1950, end_year_file=2017, variable='zos', ensemble=True)
Assi_ssh = Assi_ssh.__getitem__()
correlation_plot(Aviso_ssh, Assi_ssh, del_t=8, name_1='Aviso_ssh', name_2='Assimilation_ssh')
bias_plot(Aviso_ssh, Assi_ssh, del_t=8, name_1='Aviso_ssh', name_2='Assimilation_ssh')

#plot correlation between ocean heat content and ssts
IAP_Ohc = get_variable(path = cfg.ohc_path, start_year=196100, end_year=201600, variable='heatcontent', time_edit=False)
IAP_Ohc = IAP_Ohc.__getitem__()
IAP_Ohc = IAP_Ohc[:, ::-1, :]
correlation_plot(HadIsst, IAP_Ohc, del_t=8, name_1='HadIsst', name_2='IAP_Ohc')
