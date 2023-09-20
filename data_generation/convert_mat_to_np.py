import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import sys
from os import listdir
from os.path import isfile, join, isdir
from scipy.io import loadmat


def convert_files_from_mat_to_numpy(path):
    """
    Takes all matlab files in the given directory and converts them to npz files.
    Stores all wavelengths in a file called wavelengths.npz in the same directory.
    """
    print(path)
    if isdir(path):
        files = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f.endswith('.mat')]
        print(files)

        all_wavelengths = {}
        for f in files:
            print(f)
            print('converting {}'.format(f))
            data = loadmat(f, simplify_cells=True)
            print(data.keys())
            wavelengths = data['wavelength']
            
            
            if 'Coffee' in f:

                # determine data_name
                if 'Coffee_1' in f:
                    data_type = 'Coffee_1'
                elif 'Coffee_2' in f:
                    data_type = 'Coffee_2'
                elif 'Coffee_3' in f:
                    data_type = 'Coffee_3'
                    
                # determine sensor
                if 'SWIR3116' in f:
                    sensor = 'SWIR3116'
                    all_wavelengths['SWIR3116'] = wavelengths
                if 'SWIR3505' in f: 
                    sensor = 'SWIR3505'
                    all_wavelengths['SWIR3505'] = wavelengths
                if 'SWIR3516' in f: 
                    sensor = 'SWIR3516'
                    all_wavelengths['SWIR3516'] = wavelengths
                if 'SWIR3170' in f: 
                    sensor = 'SWIR3170'
                    all_wavelengths['SWIR3170'] = wavelengths
                if 'SWIR3122' in f: 
                    sensor = 'SWIR3122'
                    all_wavelengths['SWIR3122'] = wavelengths
                if 'Headwall' in f: 
                    sensor = 'Headwall'
                    all_wavelengths['Headwall'] = wavelengths
                else:
                    'unkown sensor'
            


            labels = data['targets']-1
            labels_cat = data['target_names']
            spectra = data['samples'].T

            print(labels_cat)
            print(labels.shape)
            print(labels_cat.shape)
            print(spectra.shape)

            print('contains {} samples with wavelength {}'.format(np.shape(spectra)[0], len(wavelengths)))
            
            np.savez('data/{}/{}_{}.npz'.format(data_type, sensor, 'Train' if 'Train' in f else 'Test'), wavelengths=wavelengths, labels=labels, spectra=spectra, label_names=labels_cat)
            
            
        if 'Coffee_1' in path:
            np.savez('data/Coffee_1/wavelenghts', SWIR3116=all_wavelengths['SWIR3116'], SWIR3505=all_wavelengths['SWIR3505'], SWIR3516=all_wavelengths['SWIR3516'])
        if 'Coffee_2' in path:
            np.savez('data/Coffee_2/wavelenghts', SWIR3116=all_wavelengths['SWIR3116'], SWIR3122=all_wavelengths['SWIR3122'])
        if 'Coffee_3' in path:
            np.savez('data/Coffee_3/wavelenghts', SWIR3505=all_wavelengths['SWIR3505'], Headwall=all_wavelengths['Headwall'])
        
       


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: <data_dir>")
    else:
        # Parse arguments
        data_dir = sys.argv[1]
        convert_files_from_mat_to_numpy(data_dir)
