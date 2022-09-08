#Loading and using all required classes to calculate leadyear timeseries

from calendar import month
import config as cfg
from preprocessing import concat
from leadyear import calculate_leadyear
import os

import tempfile
tempfile.tempdir ='/mnt/lustre02/work/uo1075/u301617/tmp'

cfg.set_args()

correlation_path = cfg.tmp_path + 'correlation/'
tmp_path = cfg.tmp_path + 'tmp/'

if not os.path.exists(cfg.plot_path):
    os.makedirs(cfg.plot_path)
if not os.path.exists(cfg.tmp_path):
    os.makedirs(cfg.tmp_path)
if not os.path.exists(correlation_path):
    os.makedirs(correlation_path)
if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)
if not os.path.exists(cfg.residual_path):
    os.makedirs(cfg.residual_path)

#save lead correlation for cfg.lead_year
if cfg.lead_years:
    lead_year = cfg.lead_years
else:
    lead_year = cfg.lead_year

ly = calculate_leadyear(cfg.start_year, cfg.end_year, lead_year=lead_year)
ly.plot()
ly.save_lead_corr()

