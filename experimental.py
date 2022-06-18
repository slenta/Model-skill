from cmath import nan
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import statsmodels.api as sm


a = np.arange(20)

pval=nan
alpha=10

signif = pval <= alpha

print(signif)
