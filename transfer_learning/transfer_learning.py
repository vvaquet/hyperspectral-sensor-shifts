import sys
import os
sys.path.append(os.getcwd())
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from standardize import preprocess_data
from utils.lookup import static_sensor_names
from utils.helper_functions import get_sensor_indices


def create_clf(model):
    if model == 'logReg-l2':
        return  LogisticRegression(penalty='l2', C=18, solver='saga', max_iter=5000)
    if model == 'logReg-l1':
        return  LogisticRegression(penalty='l1', C=16, solver='saga', max_iter=5000)
    elif model == 'randomforest':
        return RandomForestClassifier(n_jobs=4,n_estimators=500, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_depth=50, bootstrap=False)
    if model == 'svm':
        return  SVC(degree=5, kernel='poly')
    if model == 'svm-linear':
        return  SVC(kernel='linear')
    

def save_as_conf_mat(overall_results, sensor_names, f_out):

    rows = [i for i, _ in enumerate(sensor_names) for j in sensor_names]
    cols = [j for i, _ in enumerate(sensor_names) for j, _ in enumerate(sensor_names)]
    res = [overall_results[i,j] for i,j in zip(rows, cols)]

    df_cm = pd.DataFrame()
    df_cm['y'] = rows
    df_cm['x'] = cols
    df_cm['value'] = res
    df_cm.to_csv(f_out, index=False)
    print(sensor_names)


def run_experiment(f_in, f_out, data_type, model, num_folds=10, sampling_freq=1, rob_train_transversal=None, rob_features=None, rob_train_intensity=None, standardization=None, cheb=None, save_time=False, standardization_samples=-1, prepr_class=-1):
    time_used_std = []
    time_used_clf = []
    print(f_out)
    sensor_names = static_sensor_names[data_type]

    # loading the data set
    dataset = np.load(f_in)
    orig_data = dataset['spectra'].T
    all_labels = dataset['labels'].astype(int)
    all_sensors = dataset['sensors'].astype(int)
    wavelengths = np.array([i for i in range(1000, 2500)])

    subsampling_indices = [i for i in range(3, orig_data.shape[1]-3, sampling_freq)]
    orig_indices = [i for i in range(3, orig_data.shape[1]-3, 1)]
    subsampling_indices_dtw_full = [i for i in range(3, orig_data.shape[1]-3, sampling_freq)]

    if rob_features is not None:
        assert type(rob_features) == str
        rob_wavelengths = np.load(rob_features)['wavelengths']
        subsampling_indices = np.intersect1d(rob_wavelengths, [i for i in range(0, orig_data.shape[1], sampling_freq)])
    
    all_data = orig_data
    if sampling_freq != 1 and standardization != 'dtw-full':
        all_data = orig_data[:, subsampling_indices]
        wavelengths = wavelengths[subsampling_indices]
    elif sampling_freq != 1 and standardization == 'dtw-full':
        all_data = orig_data[:, orig_indices]
        wavelengths = wavelengths[orig_indices]

    if cheb is not None:
        all_data = orig_data[:, :cheb]
        orig_data = orig_data[:, :cheb]
    
    num_features = all_data.shape[1]    
    unique_labels = np.unique(all_labels)
    _, unique_sensors = get_sensor_indices(all_sensors)

    # prepare folds for crossval
    all_indices = np.array([i for i in range(all_data.shape[0])], dtype=int)
    np.random.seed(42)
    np.random.shuffle(all_indices)

    items_per_fold = int(len(all_indices)/num_folds)
    fold_indices = np.zeros((num_folds, items_per_fold), dtype=int)

    for i in range(num_folds):
        fold_indices[i, :] = all_indices[i*items_per_fold:(i+1)*items_per_fold]

    overall_results = np.empty((len(unique_sensors), len(unique_sensors), 1))
    overall_recalls = np.empty((len(unique_sensors), len(unique_sensors)*len(unique_labels), 1))
    overall_conf = np.empty((len(unique_sensors)*len(unique_labels), len(unique_sensors)*len(unique_labels), 1))
    overall_results_on_training_set = np.empty((len(unique_sensors), len(unique_sensors), 1))

    weights = np.zeros((len(unique_sensors), num_folds, len(unique_labels), num_features))

    # training and testing
    for run in range(num_folds):
        print('---{}/{}---'.format(run+1, num_folds))
        run_indices = [i for i in range(num_folds) if i!=run]
        data_train = all_data[fold_indices[run_indices, :]].reshape(((num_folds-1)*items_per_fold), num_features)
        if rob_train_transversal is not None:
            data_shift = orig_data[fold_indices[run_indices, :]].reshape(((num_folds-1)*items_per_fold), 1500)
        labels_train = all_labels[fold_indices[run_indices, :]].reshape(((num_folds-1)*items_per_fold))
        sensors_train = all_sensors[fold_indices[run_indices, :]].reshape(((num_folds-1)*items_per_fold))
        data_test = all_data[fold_indices[run, :]]
        labels_test = all_labels[fold_indices[run]]
        sensors_test = all_sensors[fold_indices[run]]

        if rob_train_transversal is not None:
            assert type(rob_train_transversal) == list
            
            for shift in rob_train_transversal:
                if standardization != 'dtw-full':
                    data_train = np.vstack((data_train, data_shift[:, subsampling_indices+(np.ones(data_train.shape[1])*shift).astype(int)]))
                else:
                    data_train = np.vstack((data_train, data_shift[:, orig_indices +(np.ones(data_train.shape[1])*shift).astype(int)]))

            labels_train = np.tile(labels_train, len(rob_train_transversal)+1)
            sensors_train = np.tile(sensors_train, len(rob_train_transversal)+1)

        if standardization is not None:
            start = time.time()
            data_train, labels_train, _ = preprocess_data(data_train, labels_train, standardization, prepr_samples=standardization_samples, prepr_class=prepr_class)
            end = time.time()
            time_used_std.append(end-start)


        if rob_train_intensity is not None:
            assert type(rob_train_intensity) == float
            data_train = np.vstack((data_train, data_train+np.ones(data_train.shape)*rob_train_intensity, data_train-np.ones(data_train.shape)*rob_train_intensity))
           
            labels_train = np.tile(labels_train, 3)
            sensors_train = np.tile(sensors_train, 3)
            
        results = np.zeros((len(unique_sensors), len(unique_sensors),1))
        results_rec = np.zeros((len(unique_sensors), len(unique_sensors)*len(unique_labels),1))
        results_conf = np.zeros((len(unique_sensors)*len(unique_labels), len(unique_sensors)*len(unique_labels),1))
        results_on_training_set = np.zeros((len(unique_sensors), len(unique_sensors),1))
        indices, unique_sensors = get_sensor_indices(sensors_train)
        test_indices, _ = get_sensor_indices(sensors_test)
        for sensor in unique_sensors:
            X_train = data_train[indices[sensor], :]
            y_train = labels_train[indices[sensor]]
   
            clf = create_clf(model=model)

            # train the model
            if standardization == 'dtw-full':
                start = time.time()
                clf.fit(X_train[:, subsampling_indices_dtw_full], y_train.ravel())
                end = time.time()
                time_used_clf.append(end-start)
            else:
                start = time.time()
                clf.fit(X_train, y_train.ravel())
                end = time.time()
                time_used_clf.append(end-start)

            # testing
            for other_sensor in unique_sensors:
                
                X_test = data_test[test_indices[other_sensor], :]
                y_test = labels_test[test_indices[other_sensor]]

                if standardization is not None:
                    start = time.time()
                    X_test, y_test, _ = preprocess_data(X_test, y_test, standardization, orig_data=X_train, wavelengths=wavelengths, prepr_samples=standardization_samples, prepr_class=prepr_class)
                    end = time.time()
                    time_used_std.append(end-start)

                if standardization == 'dtw-full':  
                    y_pred = clf.predict(X_test[:, subsampling_indices_dtw_full])
                else:
                    y_pred = clf.predict(X_test)

                results[sensor, other_sensor] = accuracy_score(y_test, y_pred)

        if run == 0:
            overall_results = results
            overall_recalls = results_rec
            overall_conf = results_conf
            overall_results_on_training_set = results_on_training_set
        else:
            overall_results = np.concatenate((overall_results, results), axis=2)
            overall_recalls = np.concatenate((overall_recalls, results_rec), axis=2)
            overall_conf = np.concatenate((overall_conf, results_conf), axis=2)
            overall_results_on_training_set = np.concatenate((overall_results_on_training_set, results_on_training_set), axis=2)

    # saving results

    print(time_used_clf)
    print(np.array(time_used_clf))
    if not os.path.exists('results/transfer/{}'.format(data_type)):
        os.makedirs('results/transfer/{}'.format(data_type))
    
    else:
        np.savez('{}.npz'.format(f_out), res=overall_results, recalls=overall_recalls, conf_mats=overall_conf, res_train=overall_results_on_training_set, clf_weights=weights, time=np.array(time_used_std))
        if save_time:
            np.savez('{}_time.npz'.format(f_out), std=np.array(time_used_std), clf=np.array(time_used_clf))

    overall_results = np.mean(overall_results, axis=2)
    
    print(overall_results)
    save_as_conf_mat(overall_results, sensor_names, '{}.csv'.format(f_out))


if __name__ == '__main__':

    num_folds = 10
    model = 'logReg-l2'
    

    # -------------------
    # how many bands are required?
    # -------------------
    for sampling_freq in [700, 600, 500, 400, 300, 200, 150, 100, 50, 40, 30, 20, 15, 10, 5, 1]:
        for data_name in ['Coffee_1', 'Coffee_2', 'Coffee_3' ]: 
            f_in = 'data/{}/{}_interpolated.npz'.format(data_name, data_name)
            f_out = 'results/transfer/{}/{}_{}_{}'.format(data_name, model, sampling_freq, num_folds)
            run_experiment(f_in, f_out, data_name, model, num_folds=num_folds, sampling_freq=sampling_freq)

    for sampling_freq in [ 7]:
        for data_name in ['Red']: 
            f_in = 'data/{}/{}_interpolated_orig.npz'.format(data_name, data_name)
            f_out = 'results/transfer/{}/{}_{}_{}'.format(data_name, model, sampling_freq, num_folds)
            run_experiment(f_in, f_out, data_name, model, num_folds=num_folds, sampling_freq=sampling_freq)

    
    # -------------------
    # how many chebs are required?
    # -------------------
    for data_name in ['Coffee_1', 'Coffee_2', 'Coffee_3' ]: 
        print([i for i in range(5, 101, 5)])
        for num_cheb in range(5, 101, 5):
            f_in = 'data/{}/{}_cheb.npz'.format(data_name, data_name)
            f_out = 'results/transfer/{}/{}_{}_{}_cheb_{}'.format(data_name, model, 1, num_folds, num_cheb)
            run_experiment(f_in, f_out, data_name, model, num_folds=num_folds, sampling_freq=1, cheb=num_cheb)

    data_name = 'Red'
    num_cheb = 30
    f_in = 'data/{}/{}_cheb.npz'.format(data_name, data_name)
    f_out = 'results/transfer/{}/{}_{}_{}_cheb_{}'.format(data_name, model, 1, num_folds, num_cheb)
    run_experiment(f_in, f_out, data_name, model, num_folds=num_folds, sampling_freq=1, cheb=num_cheb)

    # -------------------
    # model selection
    # -------------------
    sampling_freq = 6
    for data_name in ['Coffee_1', 'Coffee_2', 'Coffee_3' ]:
        for model in ['logReg-l2', 'logReg-l1', 'randomforest', 'svm-linear', 'svm']:
            f_in = 'data/{}/{}_interpolated.npz'.format(data_name, data_name)
            f_out = 'results/transfer/{}/{}_{}_{}'.format(data_name, model, sampling_freq, num_folds)
            run_experiment(f_in, f_out, data_name, model, num_folds=num_folds, sampling_freq=sampling_freq, save_time=True)

    # -------------------
    # data level methods
    # -------------------
    sampling_freq = 17
    for data_name in ['Coffee_1', 'Coffee_2', 'Coffee_3' ]:
        f_in = 'data/{}/{}_interpolated.npz'.format(data_name, data_name)
        f_out = 'results/transfer/{}/{}_{}_{}'.format(data_name, model, sampling_freq, num_folds)
        run_experiment(f_in, f_out, data_name, model, num_folds=num_folds, sampling_freq=sampling_freq)

        f_in = 'data/{}/{}_interpolated.npz'.format(data_name, data_name)
        f_out = 'results/transfer/{}/{}_{}_{}_offset_elimination'.format(data_name, model, sampling_freq, num_folds)
        run_experiment(f_in, f_out, data_name, model, num_folds=num_folds, sampling_freq=sampling_freq, standardization='mean')

        num_cheb = 30
        f_in = 'data/{}/{}_cheb.npz'.format(data_name, data_name)
        f_out = 'results/transfer/{}/{}_{}_{}_cheb_{}'.format(data_name, model, 1, num_folds, num_cheb)
        run_experiment(f_in, f_out, data_name, model, num_folds=num_folds, sampling_freq=1, cheb=num_cheb)

        for standardization in ['tca']:
            f_in = 'data/{}/{}_interpolated.npz'.format(data_name, data_name)
            f_out = 'results/transfer/{}/{}_{}_{}_{}'.format(data_name, model, sampling_freq, num_folds, standardization)
            run_experiment(f_in, f_out, data_name, model, num_folds=num_folds, sampling_freq=sampling_freq, standardization=standardization)

        for standardization in ['dtw-full', 'mm1', 'mm2', 'mm3', 'mm4']:
            f_in = 'data/{}/{}_interpolated.npz'.format(data_name, data_name)
            f_out = 'results/transfer/{}/{}_{}_{}_{}'.format(data_name, model, sampling_freq, num_folds, standardization)
            run_experiment(f_in, f_out, data_name, model, num_folds=num_folds, sampling_freq=sampling_freq, standardization=standardization)

        num_cheb = 30
        f_in = 'data/{}/{}_cheb.npz'.format(data_name, data_name)
        f_out = 'results/transfer/{}/{}_{}_{}_cheb_{}_offset_elimination'.format(data_name, model, 1, num_folds, num_cheb)
        run_experiment(f_in, f_out, data_name, model, num_folds=num_folds, sampling_freq=1, cheb=num_cheb, standardization='mean')

    # -------------------
    # robust training
    # -------------------
    sampling_freq = 17
    for data_name in ['Coffee_1','Coffee_2', 'Coffee_3']:

        f_in = 'data/{}/{}_interpolated.npz'.format(data_name, data_name)

        f_out = 'results/transfer/{}/{}_{}_{}_rob_training_transversal_3.npz'.format(data_name, model, sampling_freq, num_folds)
        run_experiment(f_in, f_out, data_name, model, num_folds=num_folds, sampling_freq=sampling_freq, rob_train_transversal=([-3, -2, -1, 1, 2, 3]))

        f_out = 'results/transfer/{}/{}_{}_{}_rob_training_intensity_0.025.npz'.format(data_name, model, sampling_freq, num_folds)
        run_experiment(f_in, f_out, data_name, model, num_folds=num_folds, sampling_freq=sampling_freq, rob_train_intensity=0.05)

        f_out = 'results/transfer/{}/{}_{}_{}_rob_features_3.npz'.format(data_name, model, sampling_freq, num_folds)
        run_experiment(f_in, f_out, data_name, model, num_folds=num_folds, sampling_freq=sampling_freq, rob_features='data/Coffee_1/robust_wavelengths_v2_1.2_eta_1.npz')

        f_out = 'results/transfer/{}/{}_{}_{}_rob_features_3_rob_training_intensity_0.025.npz'.format(data_name, model, sampling_freq, num_folds)
        run_experiment(f_in, f_out, data_name, model, num_folds=num_folds, sampling_freq=sampling_freq, rob_features='data/Coffee_1/robust_wavelengths_v2_1.2_eta_1.npz', rob_train_intensity=0.025)

        f_out = 'results/transfer/{}/{}_{}_{}_rob_features_3_rob_training_transversal_3.npz'.format(data_name, model, sampling_freq, num_folds)
        run_experiment(f_in, f_out, data_name, model, num_folds=num_folds, sampling_freq=sampling_freq, rob_features='data/Coffee_1/robust_wavelengths_v2_1.2_eta_1.npz', rob_train_transversal=[-3, -2, -1, 1, 2, 3])

    # -------------------
    # robust training + data level
    # -------------------
    sampling_freq = 17
    for data_name in ['Coffee_1','Coffee_2', 'Coffee_3']: 

        f_in = 'data/{}/{}_interpolated.npz'.format(data_name, data_name)


        for method in ['dtw-full', 'mm1', 'mm2', 'mm3', 'mm4']:

            f_out = 'results/transfer/{}/{}_{}_{}_rob_training_transversal_3_{}'.format(data_name, model, sampling_freq, num_folds, method)
            run_experiment(f_in, f_out, data_name, model, num_folds=num_folds, sampling_freq=sampling_freq, rob_train_transversal=([-3, -2, -1, 1, 2, 3]), standardization=method)


            f_out = 'results/transfer/{}/{}_{}_{}_rob_training_intensity_0.025_{}'.format(data_name, model, sampling_freq, num_folds, method)
            run_experiment(f_in, f_out, data_name, model, num_folds=num_folds, sampling_freq=sampling_freq, rob_train_intensity=0.05, standardization=method)

            f_out = 'results/transfer/{}/{}_{}_{}_rob_features_3_{}'.format(data_name, model, sampling_freq, num_folds, method)
            run_experiment(f_in, f_out, data_name, model, num_folds=num_folds, sampling_freq=sampling_freq, rob_features='data/Coffee_1/robust_wavelengths_v2_1.2_eta_1.npz', standardization=method)

       
