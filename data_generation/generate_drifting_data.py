import sys
import os
from matplotlib import pyplot as plt
sys.path.append(os.getcwd())
import numpy as np
from scipy.interpolate import interp1d
from sklearn.utils import shuffle


def generate_intensity_shifted_data(f_data, noise_type, f_out, factor=[(1,1),1]):

    print('generating ', f_out)
    
    dataset = np.load(f_data)
    data = dataset['spectra']
    labels = dataset['labels']
    wavelengths = dataset['wavelengths']

    data, labels = shuffle(data, labels, random_state=42)

    print(factor)
    f_s = np.linspace(factor[0][0], factor[0][1], len(labels))
    print(f_s)

    
    x = wavelengths[[i for i in range(0, len(wavelengths), 5)]]
    y = np.random.normal(0.02, 0.005, len(x))
    fct = interp1d(x, y, kind='cubic')
    y_smooth = fct(wavelengths)

    x_noise = wavelengths[[i for i in range(0, len(wavelengths), 5)]]
    np.random.seed(42)
    y = np.random.normal(0.001, 0.0001, len(x_noise))
    fct = interp1d(x_noise, y, kind='cubic')
    y_smooth_noise = fct(wavelengths)

    data_after = np.copy(data)

    all_noise = []
    for i in range(data_after.shape[0]):
        if noise_type == 'gaussian':
            noise = np.random.normal(y_smooth_noise, 0.002, data.shape[1])
        elif noise_type == 'weibull':
            noise = np.random.weibull(y_smooth_noise*1000, data.shape[1])/500

        shift = np.ones(data.shape[1]) * y_smooth * f_s[i] + noise * factor[1]
        data_after[i,:] = data_after[i,:] + shift
        all_noise.append(shift)

    all_noise = np.stack(all_noise)
    noise_statistics = {'min':np.min(np.abs(all_noise)), 'max':np.max(np.abs(all_noise)), 'mean':np.mean(np.abs(all_noise)), 'median':np.mean(np.abs(all_noise))}
    print('mean shift ', noise_statistics['mean'])

    
    np.savez(f_out, spectra=data_after.T, labels=labels, wavelengths=wavelengths)




def generate_transversal_shifted_data(f_interpol_data, f_shift, f_out, em_division='SWIR', factor=1):
    dataset = np.load(f_interpol_data)
    data = dataset['spectra']
    labels = dataset['labels']
    orig_wavelengths = dataset['wavelengths']

    shift_data = np.load(f_shift)
    shift = shift_data['shift']
    wavelengths = shift_data['wavelengths']

    data_before = np.empty((len(wavelengths),0)) 
    data_after = np.empty((len(wavelengths),0)) 

    print(data.shape)
    num_samples = data.shape[0]

    for i in range(num_samples):
        if i % 1000 == 0:
            print(i, '/', num_samples)
        
        fct = interp1d(orig_wavelengths, data[i, :], fill_value='extrapolate', assume_sorted=True)
        data_before = np.hstack((data_before, fct(wavelengths).reshape(-1, 1)))
        data_after =  np.hstack((data_after, fct(wavelengths-factor*shift).reshape(-1, 1)))



    total_shift = factor*shift
    noise_statistics = {'min':np.min(total_shift), 'max':np.max(total_shift), 'mean':np.mean(total_shift), 'median':np.mean(total_shift)}
    print('mean shift ', noise_statistics['mean'])
    np.savez(f_out, spectra=data_after, labels=labels, wavelengths=wavelengths, min_shift=np.min(np.abs(total_shift)), max_shift=np.max(np.abs(total_shift)), mean=np.mean(np.abs(total_shift)), median=np.mean(np.abs(total_shift)))


if __name__ == "__main__":
    f_out = 'data/SWIR3505_drifting_intensity/drift_intensity_0_8_gaussian_1.npz'
    # generate_intensity_shifted_data('data/Coffee_1/SWIR3505_Train.npz', 'gaussian', f_out, factor=[(0, 8), 1])
    generate_intensity_shifted_data('data/Coffee_1/SWIR3505_Train.npz', 'gaussian', f_out, factor=[(0, 16), 1])

    