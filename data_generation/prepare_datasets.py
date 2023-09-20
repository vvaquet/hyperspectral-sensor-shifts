import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.interpolate import interp1d
from utils.lookup import sensor_lookup_name_id, min_max_wavelength


def interpolate_file(file, VNIR=False):
    """
    Interpolates all given files with a resolution of 1nm
    """   
  
        
    print('converting {}'.format(file))

    data = np.load('{}.npz'.format(file), allow_pickle=True)
    print(list(data.keys()))
    wavelengths = data['wavelengths']
    labels = data['labels']
    spectra = data['spectra']
    if spectra.shape[1] != len(labels):
        spectra=spectra.T

    # print(wavelengths)
    print(spectra.shape)
    print(wavelengths.shape)
    spectra_interpolated = np.empty((2493-1005,0)) 
    all_wavelengths = [w for w in range(1005, 2493)]

    if VNIR:
        spectra_interpolated = np.empty((982-411,0)) 
        all_wavelengths = [w for w in range(411, 982)]
        
    for i in range(np.shape(spectra)[1]):
        if i % 1000 == 0:
            print(i, '/', np.shape(spectra)[1])
        fct = interp1d(wavelengths, spectra[:,i], assume_sorted=True, bounds_error='extrapolate')
        spectra_interpolated = np.hstack((spectra_interpolated, fct(all_wavelengths).reshape(-1, 1)))

    np.savez('{}_interpolated.npz'.format(file), labels=labels, spectra=spectra_interpolated, wavelengths=all_wavelengths)



def interpolate_files(files, sensor_lookup, data_path, dataset):
    """
    Interpolates all given files with a resolution of 1nm
    """
    print(data_path)
    
    all_interpolated_spectra = np.empty((min_max_wavelength[dataset][1]-min_max_wavelength[dataset][0],0))
    all_labels = np.empty((1,0))
    all_sensors =  np.empty((1,0))

    val_all_interpolated_spectra = np.empty((min_max_wavelength[dataset][1]-min_max_wavelength[dataset][0],0))
    val_all_labels = np.empty((1,0))
    val_all_sensors =  np.empty((1,0))
    
    for f in files:
        
        print('converting {}'.format(f))

        data = np.load(f,allow_pickle=True)
        wavelengths = data['wavelengths']
        labels = data['labels']
        label_names = data['label_names']
        spectra = data['spectra'].T
        spectra_interpolated = np.empty((min_max_wavelength[dataset][1]-min_max_wavelength[dataset][0],0)) 
        all_wavelengths = [w for w in range(min_max_wavelength[dataset][0],min_max_wavelength[dataset][1])]
        
        sensor_id = None
        for s in list(sensor_lookup.keys()):
            if f.find(s) != -1:
                sensor_id = sensor_lookup[s]
        if sensor_id == None:
            print('error')
            return
        
        for i in range(np.shape(spectra)[1]):
            if i % 1000 == 0:
                print(i, '/', np.shape(spectra)[1])
            fct = interp1d(wavelengths, spectra[:,i], assume_sorted=True)
            spectra_interpolated = np.hstack((spectra_interpolated, fct(all_wavelengths).reshape(-1, 1)))
            
        if not ('Test' in f):
            all_interpolated_spectra = np.hstack((all_interpolated_spectra, spectra_interpolated))
            all_labels = np.append(all_labels, labels)
            all_sensors = np.append(all_sensors, (np.ones(len(labels))*sensor_id))

        else:
            val_all_interpolated_spectra = np.hstack((val_all_interpolated_spectra, spectra_interpolated))
            val_all_labels = np.append(val_all_labels, labels)
            val_all_sensors = np.append(val_all_sensors, (np.ones(len(labels))*sensor_id))
        
        print(len(label_names))

    np.savez('{}/{}_interpolated.npz'.format(data_path, dataset), labels=all_labels, spectra=all_interpolated_spectra, sensors=all_sensors)
    np.savez('{}/{}_interpolated_val.npz'.format(data_path, dataset), labels=val_all_labels, spectra=val_all_interpolated_spectra, sensors=val_all_sensors)



def interpolate(dataset, data_path):
    """
    Interpolates all files in the given data_path.
    """

    sensor_lookup = sensor_lookup_name_id[dataset]
    
    files = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f)) and f.endswith('.npz') and np.sum([s in f for s in sensor_lookup.keys()])==1]
    interpolate_files(files, sensor_lookup, data_path, dataset)

def change_labels_Red():
    data_orig = np.load('data/Red/Red_interpolated_orig.npz')
    data = data_orig['spectra']
    labels = data_orig['labels']
    sensors = data_orig['sensors']
    indices = []
    indices.append(np.where(labels==4)[0])
    indices.append(np.where(labels==7)[0])
    indices.append(np.where(labels==8)[0])
    indices.append(np.where(labels==9)[0])
    indices = np.concatenate(indices)
    labels = labels[indices]-6
    labels = labels.clip(min=0)
    print(np.unique(labels))
    np.savez('data/Red/Red_interpolated.npz', spectra=data[:,indices], labels=labels, sensors=sensors[indices])




if __name__ == "__main__":
    ## prepare Coffee dataset
    # if len(sys.argv) < 1:
    #     print("Usage: <dataset> <data_dir>")
    # else:
    # Parse arguments
    # dataset = sys.argv[1]
    # data_dir = sys.argv[2]
    # interpolate(dataset, data_dir)
    # change_labels_Red()

    # for i in range(0,21):
    #     interpolate_file('data/VNIR_transversal/transversal_{}.0'.format(i), VNIR=True)

    for f_t in np.linspace(0,20,11):
        for f_s in np.linspace(0,0.25,6, dtype=float):
            # interpolate_file('data/SWIR3505_combined/combined_{}_{}'.format(round(f_t,1), round(f_s,2)))
            interpolate_file('data/VNIR_combined/combined_{}_{}'.format(round(f_t,1), round(f_s,2)), VNIR=True)
