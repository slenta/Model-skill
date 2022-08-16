#Loading and using all required classes to calculate leadyear timeseries

from calendar import month
import config as cfg
from preprocessing import concat
from leadyear import calculate_leadyear
import os

import tempfile
tempfile.tempdir ='/mnt/lustre02/work/uo1075/u301617/tmp'

cfg.set_args()

if not os.path.exists(cfg.plot_path):
    os.makedirs(cfg.plot_path)
if not os.path.exists(cfg.tmp_path):
    os.makedirs(cfg.tmp_path)

#save lead correlation for cfg.lead_year
if cfg.lead_years:
    lead_year = cfg.lead_years
else:
    lead_year = cfg.lead_year

ly = calculate_leadyear(cfg.start_year, cfg.end_year, lead_year=lead_year)
ly.plot()
ly.save_lead_corr()

