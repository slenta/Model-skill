#Loading and using all required classes to calculate leadyear timeseries

from calendar import month
import config as cfg
from preprocessing import concat
from leadyear import calculate_leadyear
from preprocessing import concat_hist

import tempfile
tempfile.tempdir ='/mnt/lustre02/work/uo1075/u301617/tmp'

cfg.set_args()

#concatenate and save historical simulations in a single file
#concate = concat_hist(cfg.hist_start_years, cfg.hist_end_years, cfg.ensemble_member, cfg.scenario_path, cfg.scenario)
#concate.concat()

#concatenate and save aviro data
#first concatenate daily data to monthly and save at tmp
#for i in range(2000, 2019):
#    print(i)
#    for j in range(1, 12):
#        month = str(j).zfill(2)
#        con = concat(cfg.aviro_path + str(i) + '/' + month + '/', 'Aviro_Ssh', variable='adt', start=str(i) + '-' + month, end=str(i + 1) + '-' + month)
#        con.concat()
#then: concatenate all monthly data over the whole timeframe
con = concat(cfg.tmp_path + 'Aviro_Ssh/', name = 'Aviro_Ssh_full', variable='var', start='2000-01', end='2019-12')
con.concat()


#save lead correlation for cfg.lead_year
if cfg.lead_years:
    lead_year = cfg.lead_years
else:
    lead_year = cfg.lead_year

#ly = calculate_leadyear(cfg.start_year, cfg.end_year, lead_year=lead_year)
#ly.plot()
#ly.save_lead_corr()

