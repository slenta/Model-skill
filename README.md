# Model-skill

## Requirements
- Python 3.7

```
pip install -r requirements.txt
```

## General
This repository is supposed to give general tools to evaluate and plot hindcast skills.
Up till now, this is implemented for decadal SST hindcasts.
The individual files have the following contents:

## Config
All general variables are defined here, in addition to some model paths and the corresponding configurations

## Preprocessing.py
Specific functions for the general preprocessing of data for skill calculation:
  - Detrending functions: Detrending input arrays depending on the specific input shape
  - Concating functions: concatenating historical and other datasets to create a single input array
  - Ensemble Means: Calculating ensemble means from an ensemble of model implementations in different NetCDF4 Files
  - Get Variable: Loading a variable from a NetCDF4 file, which is named after Dkrz regulations
  -   Here a specific lead year as well as the longitute and latitude values can additionally be extracted

## Residuals.py
Class for calculating the residual hindcast and observations for a certain model configuration and leadyear
Later, save those in a specific hdf5 format

## Leadyear.py

Calculate_leadyear class: For one leadyear: 
- Call historical, hindcast and observational data
- Choose the relevant lead year or lead years
- Call residual hindcast and observations for the leadyear in all years
- Calculate and save historical correlation, hindcast correlation, residual hindcast correlation
- Plot hindcast, historical and residual hindcast skill

Ly_series class:
- Load correlations for all leadyears from all years
- Concatenate and create series of all lead year correlations
- Plot lead year correlation series

## Plots.py

Different plotting functions to create:
- RSME plots between two variables
- Correlations between two variables
- Bias between to variables
- plot a significance mask on top of a certain variable or correlation plot

## Plotting_create_mask.py

Implementation for a specific set of plots and skill calculations

## Save_leadyear.py

An implementation to calculate and save leadyear correlations for a specific model configuration

## Decorrelation_time.py

Calculate and plot decorrelation time for a specific variable

## Corr_sig and corr_2d_ttest.py

Calculate correlation and significance for more complex ttests than just simple pearsonr

## 4d temperature_lys.py

Calculate leadyear timeseries for a 4d-temperature field for a specific hindcast configuration

