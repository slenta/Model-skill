#script to plot all wanted plots

from leadyear import ly_series
import config as cfg
from leadyear import calculate_leadyear
from preprocessing import get_variable
from decorrelation_time import decorrelation_time
from decorrelation_time import correlation_plot
import matplotlib

#matplotlib.use('Agg')
cfg.set_args()

#plot correlation for a specific lead year
#lys = calculate_leadyear(cfg.start_year, cfg.end_year, lead_year='2 9')
#lys.plot()

#plot leadyear correlation for all lead years
ly = ly_series(cfg.start_year, cfg.end_year)
ly.ly_series()

#plot decorrelation time for HadISSTs
#define threshold for decorrelation mask
threshold = 1
HadIsst = get_variable(path = cfg.observation_path, start_year = 1960, end_year = 2021)
decor = decorrelation_time(HadIsst, del_t=8, threshold=threshold)
dc, mask = decor.__getitem__()
decorrelation_time.plot()

#plot correlation between ocean heat content and ssts
IAP_Ohc = get_variable(path = cfg.ohc_path, start_year=1960, end_year=2021)
correlation_plot(HadIsst, IAP_Ohc, del_t=8, name='Ohc_4yearmean')
