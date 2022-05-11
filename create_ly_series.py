#script to load ly correlations and create ly series

from leadyear import ly_series
import config as cfg

cfg.set_args()

ly = ly_series(cfg.start_year, cfg.end_year)
ly.ly_series()