import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import statsmodels.api as sm


a = np.arange(20)

b = np.zeros(shape=(len(a) - 1, len(a)))
b[0] = a
ac = np.zeros(len(a)-1)


for i in range(0, len(a) - 1):
    b[i] = np.roll(a, i)
    ac[i] = pearsonr(a, b[i])[0]

ac_f = sm.tsa.acf(a)
dc_m = np.array(np.where(ac<1/np.e))
dc_f = np.squeeze(np.where(ac_f<1/np.e))[0]

print(dc_m.shape, dc_f)


print(ac[0])
print(np.sum(ac), np.sum(ac_f))
plt.plot(ac, label='manual')
plt.plot(sm.tsa.acf(a), label='function')
plt.legend()
plt.grid()
plt.show() 


