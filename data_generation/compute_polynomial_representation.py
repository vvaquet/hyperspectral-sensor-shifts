import sys
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy.interpolate import interp1d
import time



def chebyshev_zeros(n):
    # A chebyshev polynomial of degree n has exactly n roots
    k = list(range(0, n))
    k = np.transpose(k)
    roots = np.cos((2 * k + 1) / (2 * n) * np.pi)
    roots = np.flipud(roots)

    return roots


def resample_data(original_x, original_y, new_x):
    # Till now, only an straight linear approximation is used
    # We assume, the data is ordered in ascending order

    new_yy = np.zeros((len(new_x), len(original_y[0])))
    for i in range(len(original_y[0])):
        f = interp1d(original_x, original_y[:, i])
        new_yy[:, i] = f(new_x)

    return new_yy


def get_matrix(polynomial, degree, resample_x):
    C = np.zeros((len(resample_x), degree +1))

    if polynomial == 'cheb':
        for i in range(degree + 1):
            if i == 0:
                C[:, i] = 1
            elif i == 1:
                C[:, i] = resample_x
            else:
                C[:, i] = np.multiply(2 * resample_x, C[:, i-1]) - C[:, i - 2]
    elif polynomial == 'legendre':
        for i in range(degree + 1):
            if i == 0:
                C[:, i] = 1
            elif i == 1:
                C[:, i] = resample_x
            else:
                C[:, i] = np.multiply((2*i+1)/(i+1) * resample_x, C[:, i-1]) - i/(i+1) * C[:, i - 2]
    elif polynomial == 'herm':
        for i in range(degree + 1):
            if i == 0:
                C[:, i] = 1
            elif i == 1:
                C[:, i] = 2*resample_x
            else:
                C[:, i] = np.multiply(2*resample_x, C[:, i-1]) - 2*i * C[:, i - 2]

    return C


def calc_coefficients(data, num_coeff, polynomial):
    degree = num_coeff - 1

    # Resample the data at the roots of chebyshev polynomial of degree n+1
    data_x = np.linspace(-1, 1, num=len(data))
    resample_x = chebyshev_zeros(degree + 1)

    resample_y = resample_data(data_x, data, resample_x)

    # Compute coefficients from resampled data
    C = get_matrix(polynomial, degree, resample_x)
            
    P = (2 / (degree + 1)) * C
    P[:, 0] = 0.5 * P[:, 0]
    P = np.transpose(P)

    coefficients = np.transpose(np.dot(P, resample_y))

    return coefficients


def compute_poly_representations(f_in, f_out, degree, rep='cheb'):
    """
    Computes the polynomial representaion of a given simulated spectra stored in f_in using least squares.
    Remove wavelenghts ignores the last remove_wavelenghts wavelengths for fitting; poly_select only stores the selected coefficients.
    """
    dataset = np.load(f_in)
    print(f_in)
    print(list(dataset.keys()))
    spectra = dataset['spectra']
    labels = dataset['labels']

    data = calc_coefficients(spectra, degree, rep)
    
    data = data.T
    if 'sensors' in list(dataset.keys()):
        np.savez(f_out, spectra=data, labels=labels, sensors=dataset['sensors'], wavelengths=None)
    else:
        np.savez(f_out, spectra=data, labels=labels, wavelengths=None)


if __name__ == "__main__":
    for path in ['data/VNIR_combined','data/VNIR_transversal', 'data/VNIR_intensity']:#]:# 'data/SWIR3505_intensity_v2', 'data/SWIR3505_transversal', 'data/SWIR3505_combined']:

        files = [f for f in listdir(path) if isfile(join(path, f)) and not 'cheb' in f and '.npz' in f and not 'robust' in f]
        print(files)

        for f in files:
            f = '{}/{}'.format(path, f)
            compute_poly_representations(f, '{}_cheb.npz'.format(f[:-4]), 30, rep='cheb')


    # compute_poly_representations('data/Coffee_1/Coffee_1_interpolated.npz', 'data/Coffee_1/Coffee_1_cheb.npz', 100, rep='cheb')
    # compute_poly_representations('data/Coffee_2/Coffee_2_interpolated.npz', 'data/Coffee_2/Coffee_2_cheb.npz', 100, rep='cheb')
    # compute_poly_representations('data/Coffee_3/Coffee_3_interpolated.npz', 'data/Coffee_3/Coffee_3_cheb.npz', 100, rep='cheb')
    # compute_poly_representations('data/Red/Red_interpolated.npz', 'data/Red/Red_cheb.npz', 30, rep='cheb')
