#Loading and using all required classes to calculate leadyear timeseries

import config as cfg
from preprocessing import concat
from leadyear import calculate_leadyear

cfg.set_args()

#concatenate and save historical simulations in a single file
#concate = concat(cfg.hist_start_years, cfg.hist_end_years, cfg.ensemble_member, cfg.scenario_path, cfg.scenario)
#concate.concat()

#save lead correlation for cfg.lead_year
if cfg.lead_years:
    lead_year = cfg.lead_years
else:
    lead_year = cfg.lead_year

ly = calculate_leadyear(cfg.start_year, cfg.end_year, lead_year=lead_year)
ly.save_lead_corr()

