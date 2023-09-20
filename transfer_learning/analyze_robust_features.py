import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt


keep = 1.2
eta = 1 

# f = 'data/Coffee_1/SWIR3505_Train.npz'
f = 'data/VNIR_transversal/transversal_0.0.npz'
data_file = np.load(f)
data = data_file['spectra'].T
wavelenghts = data_file['wavelengths']
mean_spectra = np.mean(data, axis=1)

diffs = []
for i in range((eta),len(mean_spectra)-eta):
    diffs.append(abs((mean_spectra[i+eta]-mean_spectra[i-eta])/(wavelenghts[i+eta]-wavelenghts[i-eta])))


diffs = np.array(diffs)
rob_wavelengths = []
mean_diffs = np.mean(diffs)
reject = np.where(diffs > keep *mean_diffs)[0]
for i, diff in enumerate(diffs):
    if not i in reject:

        rob_wavelengths.append(i)

plt.figure()
plt.plot(diffs)

plt.plot(mean_spectra*0.01)    
plt.scatter(rob_wavelengths, np.ones(len(rob_wavelengths))*0.01)
plt.plot(np.ones(len(diffs))*mean_diffs*keep)
plt.show()
# np.savez('data/SWIR3505_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(keep, eta), wavelengths=rob_wavelengths)
np.savez('data/VNIR_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(keep, eta), wavelengths=rob_wavelengths)
