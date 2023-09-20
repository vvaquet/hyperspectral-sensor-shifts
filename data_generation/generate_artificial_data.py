import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from scipy.interpolate import interp1d


def generate_intensity_shifted_data(f_data, f_shift, f_out, factor=0.05):

    print('generating ', f_out)
    
    dataset = np.load(f_data)
    data = dataset['spectra']
    labels = dataset['labels']
    wavelengths = dataset['wavelengths']
    if data.shape[0] != len(labels):
        data = data.T

    if os.path.isfile(f_shift):
        shift = np.load(f_shift)
    else:
        x = wavelengths[[i for i in range(0, len(wavelengths), 5)]]
        y = np.random.normal(1, 0.05, len(x))
        fct = interp1d(x, y, kind='cubic', fill_value='extrapolate')
        shift = fct(wavelengths)
        shift = shift*(1/np.mean(np.abs(shift)))
        np.save(f_shift, shift)

    data_shifted = np.copy(data)

    all_noise = []
    for i in range(data_shifted.shape[0]):
        offset = np.ones(data.shape[1], dtype=float) * shift * factor
        data_shifted[i,:] = data_shifted[i,:] + offset
        all_noise.append(offset)

    all_noise = np.stack(all_noise)
    
    np.savez(f_out, spectra=data_shifted.T, labels=labels, wavelengths=wavelengths, min_shift=np.min(np.abs(all_noise)), max_shift=np.max(np.abs(all_noise)), mean_shift=np.mean(np.abs(all_noise)), median_shift=np.median(np.abs(all_noise)))


def create_fine_interpol(f_in, f_out, em_division='SWIR'):
    
        dataset = np.load(f_in)
        
        data = dataset['spectra'].T
        labels = dataset['labels']
        wavelengths = dataset['wavelengths']
       
        data = data.T
        num_samples = data.shape[0]

        if em_division == 'SWIR':
            all_wavelengths = [w/100 for w in range(100000,249900)]
        if em_division == 'VNIR':
            all_wavelengths = [w/100 for w in range(40800,98500)]

        X_interpolated = np.empty((len(all_wavelengths),0)) 
        
        for i in range(num_samples):
            if i % 100 == 0:
                print(i, '/', num_samples)
            
            fct = interp1d(wavelengths, data[i, :])
            X_interpolated = np.hstack((X_interpolated, fct(all_wavelengths).reshape(-1, 1)))

        np.savez(f_out, data=X_interpolated, labels=labels, wavelengths=all_wavelengths)


def generate_transversal_shifted_data(f_interpol_data, f_shift, f_out, factor=1, samples_per_class=None):
    print(f_out)
    dataset = np.load(f_interpol_data)
    data = dataset['spectra']
    labels = dataset['labels']
    orig_wavelengths = dataset['wavelengths']
    # print(len(orig_wavelengths))

    shift_data = np.load(f_shift)
    shift = shift_data['shift']
    shift = shift*(1/np.mean(np.abs(shift)))
    wavelengths = shift_data['wavelengths']
    # print(len(wavelengths))

    data_out = np.empty((0, len(wavelengths)))
    # print(data.shape)

    if samples_per_class is not None:
        label_subset = [4,7,8,9]
        current_labels = np.empty((1,0))
        current_data = np.empty((0, len(orig_wavelengths))) 
        print(np.unique(labels))
        for label in label_subset:
            indices = np.where(labels==label)[0]
            np.random.seed(42)
            np.random.shuffle(indices)
            # print(len(indices))
            indices = indices[0:samples_per_class]
            current_labels = np.append(current_labels, np.ones(len(indices))* label)
            current_data = np.vstack((current_data, data[indices, :]))
        data = current_data
        labels = current_labels#np.hstack(current_labels)
        labels = labels-6
        labels = labels.clip(min=0)
    # print('new shape')
    # print(data.shape)
    # print(labels.shape)
    # print(np.unique(labels))

    num_samples = data.shape[0]

    for i in range(num_samples):
        if i % 1000 == 0:
            print(i, '/', num_samples)
        
        fct = interp1d(orig_wavelengths, data[i, :], fill_value='extrapolate', assume_sorted=True) 
        # print(data_out.shape)
        data_out =  np.vstack((data_out, fct(wavelengths+factor*shift)))

    total_shift = factor*shift
    print(data_out.shape)
    
    np.savez(f_out, spectra=data_out, labels=labels, wavelengths=wavelengths, min_shift=np.min(np.abs(total_shift)), max_shift=np.max(np.abs(total_shift)), mean_shift=np.mean(np.abs(total_shift)), median_shift=np.mean(np.abs(total_shift)))


def generate_transversal_shifted_data_for_robust_training(f_interpol_data, f_out, em_division='SWIR', factor=1):
    dataset = np.load(f_interpol_data)
    data = dataset['spectra']
    print(data.shape)
    labels = dataset['labels']
    if 'wavelengths' in list(dataset.keys()):
        wavelengths = dataset['wavelengths']
        print(wavelengths.shape)
    else:
        wavelengths = [i for i in range(1000, 2500, 1)]
        data = data.T

    shift = np.ones(len(wavelengths))*factor     
    data_shifted = np.empty((len(wavelengths),0)) 

    print(data.shape)
    print(data_shifted.shape)
    num_samples = data.shape[0]

    for i in range(num_samples):
        if i % 1000 == 0:
            print(i, '/', num_samples)
        
        fct = interp1d(wavelengths, data[i, :], fill_value='extrapolate', assume_sorted=True)
        data_shifted =  np.hstack((data_shifted, fct(wavelengths+shift).reshape(-1, 1)))

    np.savez(f_out, spectra=data_shifted, labels=labels, wavelengths=wavelengths)


if __name__ == "__main__":
   
    # # artificial transversal datasets
    # for f_s in np.linspace(0,20,21):
    # #     print(f_s)
    #     # f_out =  'data/SWIR3505_transversal/transversal_{}.npz'.format(round(f_s,1))
    #     # generate_transversal_shifted_data('data/Coffee_1/SWIR3505_Train.npz', 'data/Coffee_1/basis_shift_SWIR3505.npz', f_out, factor=f_s)

    #     f_out =  'data/VNIR_transversal/transversal_{}.npz'.format(round(f_s,1))
    #     generate_transversal_shifted_data('data/VNIR0031/VNIR0031.npz', 'data/VNIR0031/basis_shift_VNIR0031.npz', f_out, factor=f_s, samples_per_class=8000)

    # # artificial intensity datasets
    # for f_s in np.linspace(0,0.25,26, dtype=float):
    #     print(f_s)
    # #     # f_out = 'data/SWIR3505_intensity_v2/intensity_{}.npz'.format(round(f_s,2))
    # #     # generate_intensity_shifted_data('data/SWIR3505_transversal/transversal_0.0.npz','data/SWIR3505_intensity_v2/intensity_shift.npy', f_out, factor=f_s)

    #     f_out = 'data/VNIR_intensity/intensity_{}.npz'.format(round(f_s,2))
    #     generate_intensity_shifted_data('data/VNIR_transversal/transversal_0.0.npz','data/VNIR_intensity/intensity_shift.npy', f_out, factor=f_s)
       
    # # enriched data for robust training
    # # for shift in [-4, -3, -2, 1, -1, 2, 3, 4]:
    # #     f_out =  'data/SWIR3505_transversal/transversal_for_robust_training_{}.npz'.format(round(shift,1))
    # #     generate_transversal_shifted_data_for_robust_training('data/SWIR3505_transversal/transversal_0.0.npz', f_out, factor=shift)
    for shift in [-3, -2, 1, -1, 2, 3]:
        f_out =  'data/VNIR_transversal/transversal_for_robust_training_{}.npz'.format(round(shift,1))
        generate_transversal_shifted_data_for_robust_training('data/VNIR_transversal/transversal_0.0.npz', f_out, factor=shift)



    # # # artificial datasets containing both shifts
    # for f_t in np.linspace(0,20,11):
    #     for f_s in np.linspace(0,0.25,6, dtype=float):
        
    #         # f_out =  'data/SWIR3505_combined/combined_{}_{}.npz'.format(round(f_t,1), round(f_s,2))
    #         # generate_intensity_shifted_data('data/SWIR3505_transversal/transversal_{}.npz'.format(round(f_t,1)), 'data/SWIR3505_intensity/intensity_shift_for_combined.npy', f_out, factor=f_s)
    #         f_out =  'data/VNIR_combined/combined_{}_{}.npz'.format(round(f_t,1), round(f_s,2))
    #         generate_intensity_shifted_data('data/VNIR_transversal/transversal_{}.npz'.format(round(f_t,1)), 'data/VNIR_intensity/intensity_shift_for_combined.npy', f_out, factor=f_s)