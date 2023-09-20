import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from scipy.stats import trim_mean
from scipy.stats.mstats import winsorize
from scipy import optimize
from dtw import dtw, warp
from transfertools.models import TCA
from sklearn.utils import shuffle

from utils.helper_functions import get_label_indices


def standardized_moment(X, k):
    return np.mean(((X - np.mean(X))/np.std(X))**k)


def transversal(wavelength, wavelengths, X):

    if wavelength < wavelengths[0]:
        return X[:, 0]
    elif wavelength > wavelengths[-1]:
        return X[:, -1]
    else:
        id0, id1 = abs(wavelengths-wavelength).argsort()[0:2]

        if id0 > id1:
            id = id0
            id0 = id1
            id1 = id
        interpolated = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            interpolated[i] = X[i, id0] + (wavelength- wavelengths[id0]) * ((X[i, id1] -  X[i, id0])/(wavelengths[id1] - wavelengths[id0]))        
        return interpolated


def pds(X_source, X_target, window, X, tol=0.001):

    num_samples = X_source.shape[0]
    num_wavelengths = X_source.shape[1]
    b = np.zeros((num_wavelengths, num_wavelengths))

    for i in range(num_wavelengths):
        j = int((window-1)/2)
        if i <= j:
            b[0:i+j, i] = np.matmul(np.linalg.pinv(X_target[:,0:i+j], rcond=tol), X_source[:,i].reshape(-1, 1)).flatten()
        elif i+j > num_samples:
            b[i-j:, i] = np.matmul(np.linalg.pinv(X_target[:, i-j:], rcond=tol), X_source[:,i].reshape(-1, 1)).flatten()
        else:
            b[i-j:i+j, i] = np.matmul(np.linalg.pinv(X_target[:, i-j:i+j], rcond=tol), X_source[:,i].reshape(-1, 1)).flatten()

    return np.matmul(X,b)


def moment_matching(data, labels, data_orig, preprocessing, wavelengths, f_out_params):
    target_moments = np.mean(data_orig, axis=0)
    data_standardized = np.zeros((data.shape[0], data.shape[1]))
    shift_trans = []
    lambda_mm2 = 0.0001 * 0.005
    lambda_mm4 = 0.0001 * 0.05 *0.3
    lambda_mm4a = 0.0001 * 0.05 *0.7
    lambda_mm3 = 0.0001 * 0.005



    set_up = {
                'mm1' : {'target' : lambda x: 1000*(np.mean(transversal(wavelengths[wl] + x[0], wavelengths, data))-target_moments[wl])**2,
                                        'constraints':None, 
                                        'bounds':None,
                                        'solver':'Nelder-Mead',
                                        'start_value':[0]},
                'mm2': {'target':lambda x: 1000*(lambda_mm2 *x[0]**2 + (1-lambda_mm2)*(np.mean(transversal(wavelengths[wl] + x[0], wavelengths, data))-target_moments[wl])**2),
                                        'constraints':None, 
                                        'bounds':None,
                                        'solver':'Nelder-Mead',
                                        'start_value':[0]},
                'mm3': {'target':lambda x: 1000*(lambda_mm3*(last_shift-x[0])**2 + (1-lambda_mm3)*(np.mean(transversal(wavelengths[wl] + x[0], wavelengths, data))-target_moments[wl])**2),
                                        'constraints':None, 
                                        'bounds':None,
                                        'solver':'Nelder-Mead',
                                        'start_value':[0]},
                'mm4': {'target':lambda x: 1000*(lambda_mm4 *x[0]**2 + lambda_mm4a*(last_shift-x[0])**2 + (1-lambda_mm4-lambda_mm4a)*(np.mean(transversal(wavelengths[wl] + x[0], wavelengths, data))-target_moments[wl])**2),
                                        'constraints':None, 
                                        'bounds':None,
                                        'solver':'Nelder-Mead',
                                        'start_value':[0]},
                
            }
    
    fails = 0
    prev = 0
    last_shift = 0
    
    
    last_shift = 0
    data_standardized = np.zeros((data.shape[0], data.shape[1]))

    shift = []
    prev_trans_shift=0
    for wl in range(data.shape[1]):
        
        result = optimize.minimize(set_up[preprocessing]['target'], [prev_trans_shift], constraints=set_up[preprocessing]['constraints'], bounds=set_up[preprocessing]['bounds'], method='SLSQP')#set_up[preprocessing]['solver'])
        params = result.x

        last_shift = params[0]

        if not result.success:
            print('optimization failed')
            prev_trans_shift = 0
            shift.append(None)

        else:
            prev_trans_shift=params[0]
            shift.append(params[0])
        data_standardized[:, wl] = transversal(wavelengths[wl]+ params[0], wavelengths, data).T

    if f_out_params is not None:
        np.savez(f_out_params, trans=np.array(shift))
    
    return data_standardized, labels, None


def preprocess_data(data, labels, preprocessing, prepr_samples=-1, orig_data=None, trim_mean_cut=0.05, classwise_sub=False, prepr_class=-1,sensor_id=None, date_id=None, feature_subset='', target_moments=None, wavelengths=None, f_out_params=None, args=None):
    
    # select subset for preprocessing
    if prepr_samples == -1:
        data_subset = data
        label_subset = labels
    else:
        indices_labels, unique_labels = get_label_indices(labels)
        samples_per_class = int(prepr_samples/len(unique_labels))
        data_subset = []
        label_subset = []
        print(data.shape)
        for c in unique_labels:
            data_shuf, labels_shuf = shuffle(data[indices_labels[c],:], labels[indices_labels[c]])
            data_subset.append(data_shuf[:samples_per_class, :])
            label_subset.append(labels_shuf[:samples_per_class])
        
        data_subset = np.vstack(data_subset)
        label_subset = np.hstack(label_subset)

    # select specific class for preprocessing
    if prepr_class != -1:
        label_indices, unique_labels = get_label_indices(label_subset)
        data_subset = data_subset[label_indices[prepr_class], :]
        label_subset = label_subset[label_indices[prepr_class]]

    # standardization mechanisms
    if preprocessing == 'pds':
        if orig_data is None:
            return data, labels, None
        else:
            res = pds(orig_data[0:data.shape[0], :], data, 3, data)
            return res, labels, None

    if 'tca' in preprocessing:
        if orig_data is None:
            return data, labels, None
        else:
            if preprocessing == 'tca':
                transfor = TCA(mu=0.8, kernel_type='polynomial', degree=3, scaling='standard', n_components=10)
            return_data = [None, None]
            orig = shuffle(orig_data)
            shifted = shuffle(data)
            transfor.fit(orig[0:3000], shifted[0:3000])
            return_data[0] = transfor.transfer(orig_data)
            return_data[1] = transfor.transfer(data)
            return return_data, labels, None
    if 'dtw' in preprocessing:
        if orig_data is None:
            return data, labels, None
        else:
            mean_after = np.mean(data, axis=0)
            mean_before = np.mean(orig_data, axis=0)
            alignment = dtw(mean_before, mean_after, keep_internals=True)
            shift_first = warp(alignment, index_reference=True)
        
            data_warped = np.hstack((data[:, shift_first], data[:, shift_first[-1]].reshape(-1,1)))

            shift = []
            for i in range(len(wavelengths)-1):
                shift.append(wavelengths[shift_first[i]]-wavelengths[i])
            
            print(f_out_params)
            np.savez('{}.npz'.format(f_out_params), trans=np.array(shift))
            return data_warped, labels, shift

    if 'mm' in preprocessing:
        if orig_data is None:
            return data, labels, None
        else:
            return moment_matching(data, labels, orig_data, preprocessing, wavelengths, f_out_params)

    else:
        normalization = compute_normalization(preprocessing, data_subset)
        if preprocessing in ['std', 'winsorized-std', 'mean-ad']:
            return data / normalization, labels, normalization
        else:
            return data - normalization, labels, normalization


def compute_normalization(preprocessing, data_subset, trim_mean_cut=0.05):
    if preprocessing == 'mean':
        return np.mean(data_subset, axis=0)
    if preprocessing == 'winsorized-mean':
        return np.mean(winsorize(data_subset, axis=0, limits=[0.05, 0.05]), axis=0)
    if preprocessing == 'trim-mean':
        return trim_mean(data_subset, proportiontocut=trim_mean_cut, axis=0)
    if preprocessing == 'median':
        return np.median(data_subset, axis=0)
    if preprocessing == 'std':
        return  np.std(data_subset, axis=0, ddof=2)
    if preprocessing == 'winsorized-std':
        return  np.std(winsorize(data_subset, axis=0), axis=0)
    if preprocessing == 'mean-ad':
        return  np.mean(data_subset - np.median(data_subset, axis=0))

