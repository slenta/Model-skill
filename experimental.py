from cmath import nan
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import statsmodels.api as sm
import xarray as xr
import cdo

cdo = cdo.Cdo()

test = "./tmp/MIROC6/test.nc"
ifile = "./tmp/MIROC6/pi_contol_MIROC6_3200_3999.nc"
ofile = "./tmp/MIROC6/pi_control_thetao_3200_3999.nc"

cdo.selvar("thetao", input=ifile, output=ofile)


ds = xr.load_dataset(ofile)
print(ds)
