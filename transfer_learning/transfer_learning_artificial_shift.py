import sys
import os
sys.path.append(os.getcwd())
from standardize import preprocess_data
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def create_clf(model):
    if model == 'logReg-l2':
        return  LogisticRegression(penalty='l2', C=18, solver='saga', max_iter=5000)
    if model == 'logReg-l1':
        return  LogisticRegression(penalty='l1', C=16, solver='saga', max_iter=5000)
    if model == 'logReg-elasticnet':
        return  LogisticRegression(penalty='elasticNet', C=18, solver='saga', max_iter=5000)
    elif model == 'randomforest':
        return RandomForestClassifier(n_jobs=4,n_estimators=500, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_depth=50, bootstrap=False)
    if model == 'svm':
        return  SVC(degree=5, kernel='poly')
    if model == 'svm-linear':
        return  SVC(kernel='linear')


def run_experiment(f_in_train, f_in_tests, f_outs, model, num_folds=10, sampling_freq=1, rob_train_transversal=None, rob_features=None, rob_train_intensity=None, standardization=None, standardization_samples=-1, prepr_class=-1):
    
    print('training on {} with standardization {}'.format(f_in_train, standardization))

    # loading the data set
    dataset = np.load(f_in_train, allow_pickle=True)
    all_data = dataset['spectra']
    all_labels = dataset['labels'].astype(int)
    wavelengths = dataset['wavelengths'] 
    print(all_data.shape)
    if all_data.shape[0] != len(all_labels):
        all_data = all_data.T 

    # subsampling indices
    subsampling_indices = [i for i in range(0, all_data.shape[1], sampling_freq)]
    print(len(subsampling_indices))
    print(all_data.shape[1])
    print(sampling_freq)

    # adapt selected features according to robust feature selection
    if rob_features is not None:
        assert type(rob_features) == str
        rob_wavelengths = np.load(rob_features)['wavelengths']
        subsampling_indices = np.intersect1d(rob_wavelengths, [i for i in range(0, all_data.shape[1], sampling_freq)])

    # perform subsampling
    if sampling_freq != 1 and standardization != 'dtw-full':
        all_data = all_data[:,subsampling_indices]
        wavelengths = wavelengths[subsampling_indices]
    print(all_data.shape)

    # load additional data for robust training
    if rob_train_transversal is not None:
        assert type(rob_train_transversal) == tuple
        f_in_rob = rob_train_transversal[0]
        rob_shifts = rob_train_transversal[1]

        shifted_data = {shift : np.load('{}_{}.npz'.format(f_in_rob, round(shift,1)))['spectra'].T for shift in rob_shifts}
        print(shifted_data[1].shape)
        if standardization != 'dtw-full':
            for shift in rob_shifts:
                shifted_data[shift] = shifted_data[shift][:,subsampling_indices]
        

    # prepare folds for crossval
    all_indices = np.array([i for i in range(all_data.shape[0])], dtype=int)
    np.random.seed(42)
    np.random.shuffle(all_indices)

    items_per_fold = int(len(all_indices)/num_folds)
    fold_indices = np.zeros((num_folds, items_per_fold), dtype=int)

    for i in range(num_folds):
        fold_indices[i, :] = all_indices[i*items_per_fold:(i+1)*items_per_fold]

    # training and testing
    for run in range(num_folds):
        print('---{}/{}---'.format(run+1, num_folds))
        run_indices = [i for i in range(num_folds) if i!=run]
        train_indices= fold_indices[run_indices, :].reshape((num_folds-1)*items_per_fold)
        X_train = all_data[train_indices, :]
        y_train = all_labels[train_indices]

        if rob_train_transversal is not None:
            data_robust_training = [shifted_data[s][train_indices] for s in rob_shifts]
            data_robust_training = np.vstack(data_robust_training)
            labels_robust_training = np.tile(y_train, len(rob_shifts))

            print(X_train.shape)
            print(data_robust_training.shape)
            X_train = np.vstack((X_train, data_robust_training))
            y_train = np.hstack((y_train, labels_robust_training))

            print(X_train.shape)
            print(y_train.shape)

        if standardization is not None:
            X_train, y_train, _ = preprocess_data(X_train, y_train, standardization, prepr_samples=standardization_samples, prepr_class=prepr_class)

        if rob_train_intensity is not None:
            X_train = np.vstack((X_train, X_train+np.ones(X_train.shape)*rob_train_intensity, X_train-np.ones(X_train.shape)*rob_train_intensity))
            y_train = np.hstack((y_train, y_train, y_train))


        clf = create_clf(model=model)

        if standardization != 'tca':
            # train the model
            if standardization == 'dtw-full':
                clf.fit(X_train[:,subsampling_indices], y_train.ravel())
            else:
                clf.fit(X_train, y_train.ravel())       

        # testing
        for shift_s, (f_in_test, f_out) in enumerate(zip(f_in_tests, f_outs)):
            print(f_in_test, f_out)
            
            dataset = np.load(f_in_test)
            X_test = dataset['spectra']
            y_test = dataset['labels'].astype(int)

            if X_test.shape[0] != len(y_test):
                X_test = X_test.T

            # test fold only
            X_test = X_test[fold_indices[run, :]]
            y_test = y_test[fold_indices[run]]

            #subsampling
            if sampling_freq != 1 and standardization != 'dtw-full':
                X_test = X_test[:, subsampling_indices]

            if standardization is not None:
                X_test, y_test, _ = preprocess_data(X_test, y_test, standardization, orig_data=X_train, wavelengths=wavelengths, prepr_samples=standardization_samples, prepr_class=prepr_class, f_out_params='{}_shift_{}'.format(f_out, run))
            
            if standardization == 'tca':
                print(X_test[0].shape)
                print(X_test[1].shape)
                clf.fit(X_test[0], y_train.ravel())
                y_pred = clf.predict(X_test[1])

            else:
                if standardization == 'dtw-full':
                    y_pred = clf.predict(X_test[:, subsampling_indices])
                else:
                    y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            print(acc)

            np.savez('{}_{}.npz'.format(f_out, run), acc=acc)


def summarize_results(f_ins, num_folds, f_out, shifts=None):
    all_res = []
    res_per_run = []
    for f in f_ins:
        res = []
        for i in range(num_folds):
            print('{}_{}.npz'.format(f, i))
            res.append(np.load('{}_{}.npz'.format(f, i))['acc'])
            res_per_run.append(np.load('{}_{}.npz'.format(f, i))['acc'])
        mean_acc =np.mean(np.array(res))
        print(mean_acc)
        all_res.append(mean_acc)
    np.savez('{}.npz'.format(f_out), res=all_res)
    df = pd.DataFrame(all_res, columns=['acc'])
    df_all = pd.DataFrame(res_per_run, columns=['acc'])
    if shifts is not None:
        df['shift'] = shifts
    print(df.head())
    df.to_csv('{}.csv'.format(f_out))
    df_all.to_csv('{}_all.csv'.format(f_out))
    print(f_out)



def perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization=None, rob_train_intensity=None, rob_train_transversal=None, rob_features=None, standardization_samples=-1, prepr_class=-1):
    run_experiment(f_in_train=f_in_train, f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, standardization=standardization, rob_train_intensity=rob_train_intensity, rob_train_transversal=rob_train_transversal, rob_features=rob_features, standardization_samples=standardization_samples, prepr_class=prepr_class)
    summarize_results(f_outs, num_folds, f_out_summarized, shift_range)
        
if __name__ == '__main__':

    num_folds = 10
    sampling_freq = 3
    model = 'logReg-l2'
    
    # ------------------------------------------------------------------------------
    # What is the effect of the shift on model accuracy / no intervention
    # ------------------------------------------------------------------------------

    #transversal
    shift_range = np.linspace(0,20,21)
    f_in_tests = ['data/SWIR3505_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
    f_outs = ['results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_{}'.format(round(f_s), model, sampling_freq, num_folds) for f_s in shift_range]
    f_in_train = 'data/SWIR3505_transversal/transversal_0.0.npz'
    f_out_summarized = 'results/transfer/SWIR3505_transversal/transversal_{}_{}_{}'.format(model, sampling_freq, num_folds)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq)
    
    # intensity
    shift_range = np.linspace(0,0.25,26)
    f_in_tests = ['data/SWIR3505_intensity_v2/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
    f_outs = ['results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_{}'.format(round(f_s,2), model, sampling_freq, num_folds) for f_s in shift_range]
    f_in_train = 'data/SWIR3505_intensity_v2/intensity_0.0.npz'
    f_out_summarized = 'results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}'.format(model, sampling_freq, num_folds)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq)
    
    #  combined shift
    shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
    f_in_tests = ['data/SWIR3505_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
    f_outs = ['results/transfer/SWIR3505_combined/combined_{}_{}_{}_{}_{}'.format(f_t, f_s, model, sampling_freq, num_folds) for (f_t, f_s) in shift_range]
    run_experiment(f_in_train='data/SWIR3505_combined/combined_0.0_0.0.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq)
    for f in f_outs:
        summarize_results([f], num_folds, f)

    sampling_freq = 2
    #transversal
    shift_range = np.linspace(0,20,21)
    f_in_tests = ['data/VNIR_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
    f_outs = ['results/transfer/VNIR_transversal/transversal_{}_{}_{}_{}'.format(round(f_s), model, sampling_freq, num_folds) for f_s in shift_range]
    f_in_train = 'data/VNIR_transversal/transversal_0.0.npz'
    f_out_summarized = 'results/transfer/VNIR_transversal/transversal_{}_{}_{}'.format(model, sampling_freq, num_folds)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq)
    
    # intensity
    shift_range = np.linspace(0,0.25,26)
    f_in_tests = ['data/VNIR_intensity/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
    f_outs = ['results/transfer/VNIR_intensity/intensity_{}_{}_{}_{}'.format(round(f_s,2), model, sampling_freq, num_folds) for f_s in shift_range]
    f_in_train = 'data/VNIR_intensity/intensity_0.0.npz'
    f_out_summarized = 'results/transfer/VNIR_intensity/intensity_{}_{}_{}'.format(model, sampling_freq, num_folds)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq)
    
    #  combined shift
    shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
    f_in_tests = ['data/VNIR_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
    f_outs = ['results/transfer/VNIR_combined/combined_{}_{}_{}_{}_{}'.format(f_t, f_s, model, sampling_freq, num_folds) for (f_t, f_s) in shift_range]
    run_experiment(f_in_train='data/VNIR_combined/combined_0.0_0.0.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq)
    for f in f_outs:
        summarize_results([f], num_folds, f)


    # ------------------------------------------------------------------------------
    # Reversing intensity shift
    # ------------------------------------------------------------------------------

    # ------------------------
    # Chebyshev
    # ------------------------

    # transversal
    shift_range = np.linspace(0,20,21)
    f_in_tests = ['data/SWIR3505_transversal/transversal_{}_cheb.npz'.format(round(f_s, 1)) for f_s in shift_range]
    f_outs = ['results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_{}_cheb'.format(round(f_s), model, 1, num_folds) for f_s in shift_range]
    f_in_train = 'data/SWIR3505_transversal/transversal_0.0_cheb.npz'
    f_out_summarized = 'results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_cheb'.format(model, 1, num_folds)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, 1)
    
    # intensity
    shift_range = np.linspace(0,0.25,26)
    f_in_tests = ['data/SWIR3505_intensity_v2/intensity_{}_cheb.npz'.format(round(f_s, 2)) for f_s in shift_range]
    f_outs = ['results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_{}_cheb'.format(round(f_s,2), model, 1, num_folds) for f_s in shift_range]
    f_in_train = 'data/SWIR3505_intensity_v2/intensity_0.0_cheb.npz'
    f_out_summarized = 'results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_cheb'.format(model, 1, num_folds)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, 1)
    
    #  combined shift
    shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
    f_in_tests = ['data/SWIR3505_combined/combined_{}_{}_cheb.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
    f_outs = ['results/transfer/SWIR3505_combined/combined_{}_{}_{}_{}_{}_cheb'.format(f_t, f_s, model, 1, num_folds) for (f_t, f_s) in shift_range]
    run_experiment(f_in_train='data/SWIR3505_combined/combined_0.0_0.0_cheb.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=1)
    for f in f_outs:
        summarize_results([f], num_folds, f)

    sampling_freq = 2
    # transversal
    shift_range = np.linspace(0,20,21)
    f_in_tests = ['data/VNIR_transversal/transversal_{}_cheb.npz'.format(round(f_s, 1)) for f_s in shift_range]
    f_outs = ['results/transfer/VNIR_transversal/transversal_{}_{}_{}_{}_cheb'.format(round(f_s), model, 1, num_folds) for f_s in shift_range]
    f_in_train = 'data/VNIR_transversal/transversal_0.0_cheb.npz'
    f_out_summarized = 'results/transfer/VNIR_transversal/transversal_{}_{}_{}_cheb'.format(model, 1, num_folds)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, 1)
    
    # intensity
    shift_range = np.linspace(0,0.25,26)
    f_in_tests = ['data/VNIR_intensity/intensity_{}_cheb.npz'.format(round(f_s, 2)) for f_s in shift_range]
    f_outs = ['results/transfer/VNIR_intensity/intensity_{}_{}_{}_{}_cheb'.format(round(f_s,2), model, 1, num_folds) for f_s in shift_range]
    f_in_train = 'data/VNIR_intensity/intensity_0.0_cheb.npz'
    f_out_summarized = 'results/transfer/VNIR_intensity/intensity_{}_{}_{}_cheb'.format(model, 1, num_folds)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, 1)
    
    #  combined shift
    shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
    f_in_tests = ['data/VNIR_combined/combined_{}_{}_cheb.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
    f_outs = ['results/transfer/VNIR_combined/combined_{}_{}_{}_{}_{}_cheb'.format(f_t, f_s, model, 1, num_folds) for (f_t, f_s) in shift_range]
    run_experiment(f_in_train='data/VNIR_combined/combined_0.0_0.0_cheb.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=1)
    for f in f_outs:
        summarize_results([f], num_folds, f)
    
    # ------------------------
    # offset elimination
    # ------------------------

    #transversal
    shift_range = np.linspace(0,20,21)
    f_in_tests = ['data/SWIR3505_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
    f_outs = ['results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_{}_offset_elim'.format(round(f_s), model, sampling_freq, num_folds) for f_s in shift_range]
    f_in_train = 'data/SWIR3505_transversal/transversal_0.0.npz'
    f_out_summarized ='results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_offset_elim'.format(model, sampling_freq, num_folds)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization='mean')
    
    # intensity
    shift_range = np.linspace(0,0.25,26)
    f_in_tests = ['data/SWIR3505_intensity_v2/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
    f_outs = ['results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_{}_offset_elim'.format(round(f_s,2), model, sampling_freq, num_folds) for f_s in shift_range]
    f_in_train = 'data/SWIR3505_intensity_v2/intensity_0.0.npz'
    f_out_summarized = 'results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_offset_elim'.format(model, sampling_freq, num_folds)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization='mean')

    #  combined shift
    shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
    f_in_tests = ['data/SWIR3505_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
    f_outs = ['results/transfer/SWIR3505_combined/combined_{}_{}_{}_{}_{}_offset_elim'.format(f_t, f_s, model, sampling_freq, num_folds) for (f_t, f_s) in shift_range]
    run_experiment(f_in_train='data/SWIR3505_combined/combined_0.0_0.0.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, standardization='mean')
    for f in f_outs:
        print(f)
        summarize_results([f], num_folds, f)

    sampling_freq = 2
    #transversal
    shift_range = np.linspace(0,20,21)
    f_in_tests = ['data/VNIR_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
    f_outs = ['results/transfer/VNIR_transversal/transversal_{}_{}_{}_{}_offset_elim'.format(round(f_s), model, sampling_freq, num_folds) for f_s in shift_range]
    f_in_train = 'data/VNIR_transversal/transversal_0.0.npz'
    f_out_summarized ='results/transfer/VNIR_transversal/transversal_{}_{}_{}_offset_elim'.format(model, sampling_freq, num_folds)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization='mean')
    
    # intensity
    shift_range = np.linspace(0,0.25,26)
    f_in_tests = ['data/VNIR_intensity/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
    f_outs = ['results/transfer/VNIR_intensity/intensity_{}_{}_{}_{}_offset_elim'.format(round(f_s,2), model, sampling_freq, num_folds) for f_s in shift_range]
    f_in_train = 'data/VNIR_intensity/intensity_0.0.npz'
    f_out_summarized = 'results/transfer/VNIR_intensity/intensity_{}_{}_{}_offset_elim'.format(model, sampling_freq, num_folds)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization='mean')

    #  combined shift
    shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
    f_in_tests = ['data/VNIR_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
    f_outs = ['results/transfer/VNIR_combined/combined_{}_{}_{}_{}_{}_offset_elim'.format(f_t, f_s, model, sampling_freq, num_folds) for (f_t, f_s) in shift_range]
    run_experiment(f_in_train='data/VNIR_combined/combined_0.0_0.0.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, standardization='mean')
    for f in f_outs:
        print(f)
        summarize_results([f], num_folds, f)

    # ------------------------
    # offset elimination + cheb
    # ------------------------

    #transversal
    shift_range = np.linspace(0,20,21)
    f_in_tests = ['data/SWIR3505_transversal/transversal_{}_cheb.npz'.format(round(f_s, 1)) for f_s in shift_range]
    f_outs = ['results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_{}_cheb_offset_elim'.format(round(f_s), model, 1, num_folds) for f_s in shift_range]
    f_in_train = 'data/SWIR3505_transversal/transversal_0.0_cheb.npz'
    f_out_summarized ='results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_cheb_offset_elim'.format(model, 1, num_folds)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, 1, standardization='mean')
    
    # intensity
    shift_range = np.linspace(0,0.25,26)
    f_in_tests = ['data/SWIR3505_intensity_v2/intensity_{}_cheb.npz'.format(round(f_s, 2)) for f_s in shift_range]
    f_outs = ['results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_{}_cheb_offset_elim'.format(round(f_s,2), model, 1, num_folds) for f_s in shift_range]
    f_in_train = 'data/SWIR3505_intensity_v2/intensity_0.0_cheb.npz'
    f_out_summarized = 'results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_cheb_offset_elim'.format(model, 1, num_folds)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, 1, standardization='mean')

    #  combined shift
    shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
    f_in_tests = ['data/SWIR3505_combined/combined_{}_{}_cheb.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
    f_outs = ['results/transfer/SWIR3505_combined/combined_{}_{}_{}_{}_{}_cheb_offset_elim'.format(f_t, f_s, model, 1, num_folds) for (f_t, f_s) in shift_range]
    run_experiment(f_in_train='data/SWIR3505_combined/combined_0.0_0.0_cheb.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=1, standardization='mean')
    for f in f_outs:
        print(f)
        summarize_results([f], num_folds, f)

    sampling_freq = 2
    #transversal
    shift_range = np.linspace(0,20,21)
    f_in_tests = ['data/VNIR_transversal/transversal_{}_cheb.npz'.format(round(f_s, 1)) for f_s in shift_range]
    f_outs = ['results/transfer/VNIR_transversal/transversal_{}_{}_{}_{}_cheb_offset_elim'.format(round(f_s), model, 1, num_folds) for f_s in shift_range]
    f_in_train = 'data/VNIR_transversal/transversal_0.0_cheb.npz'
    f_out_summarized ='results/transfer/VNIR_transversal/transversal_{}_{}_{}_cheb_offset_elim'.format(model, 1, num_folds)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, 1, standardization='mean')
    
    # intensity
    shift_range = np.linspace(0,0.25,26)
    f_in_tests = ['data/VNIR_intensity/intensity_{}_cheb.npz'.format(round(f_s, 2)) for f_s in shift_range]
    f_outs = ['results/transfer/VNIR_intensity/intensity_{}_{}_{}_{}_cheb_offset_elim'.format(round(f_s,2), model, 1, num_folds) for f_s in shift_range]
    f_in_train = 'data/VNIR_intensity/intensity_0.0_cheb.npz'
    f_out_summarized = 'results/transfer/VNIR_intensity/intensity_{}_{}_{}_cheb_offset_elim'.format(model, 1, num_folds)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, 1, standardization='mean')

    #  combined shift
    shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
    f_in_tests = ['data/VNIR_combined/combined_{}_{}_cheb.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
    f_outs = ['results/transfer/VNIR_combined/combined_{}_{}_{}_{}_{}_cheb_offset_elim'.format(f_t, f_s, model, 1, num_folds) for (f_t, f_s) in shift_range]
    run_experiment(f_in_train='data/VNIR_combined/combined_0.0_0.0_cheb.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=1, standardization='mean')
    for f in f_outs:
        print(f)
        summarize_results([f], num_folds, f)


    # ------------------------
    # TCA
    # ------------------------

    for tca_type in ['tca']:
        #transversal
        shift_range = np.linspace(0,20,21)
        f_in_tests = ['data/SWIR3505_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
        f_outs = ['results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_{}_{}'.format(round(f_s), model, sampling_freq, num_folds, tca_type) for f_s in shift_range]
        f_in_train = 'data/SWIR3505_transversal/transversal_0.0.npz'
        f_out_summarized = 'results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_{}'.format(model, sampling_freq, num_folds, tca_type)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization=tca_type)
       

        # intensity
        shift_range = np.linspace(0,0.25,26)
        f_in_tests = ['data/SWIR3505_intensity_v2/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
        f_outs = ['results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_{}_{}'.format(round(f_s,2), model, sampling_freq, num_folds, tca_type) for f_s in shift_range]
        f_in_train = 'data/SWIR3505_intensity_v2/intensity_0.0.npz'
        f_out_summarized = 'results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_{}'.format(model, sampling_freq, num_folds, tca_type)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization=tca_type)
        
        #  combined shift
        shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
        f_in_tests = ['data/SWIR3505_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
        f_outs = ['results/transfer/SWIR3505_combined/combined_{}_{}_{}_{}_{}_{}'.format(f_t, f_s, model, sampling_freq, num_folds, tca_type) for (f_t, f_s) in shift_range]
        run_experiment(f_in_train='data/SWIR3505_combined/combined_0.0_0.0.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, standardization=tca_type)
        for f in f_outs:
            print(f)
            summarize_results([f], num_folds, f)

    sampling_freq = 2
    for tca_type in ['tca']:
        #transversal
        shift_range = np.linspace(0,20,21)
        f_in_tests = ['data/VNIR_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
        f_outs = ['results/transfer/VNIR_transversal/transversal_{}_{}_{}_{}_{}'.format(round(f_s), model, sampling_freq, num_folds, tca_type) for f_s in shift_range]
        f_in_train = 'data/VNIR_transversal/transversal_0.0.npz'
        f_out_summarized = 'results/transfer/VNIR_transversal/transversal_{}_{}_{}_{}'.format(model, sampling_freq, num_folds, tca_type)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization=tca_type)
       

        # intensity
        shift_range = np.linspace(0,0.25,26)
        f_in_tests = ['data/VNIR_intensity/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
        f_outs = ['results/transfer/VNIR_intensity/intensity_{}_{}_{}_{}_{}'.format(round(f_s,2), model, sampling_freq, num_folds, tca_type) for f_s in shift_range]
        f_in_train = 'data/VNIR_intensity/intensity_0.0.npz'
        f_out_summarized = 'results/transfer/VNIR_intensity/intensity_{}_{}_{}_{}'.format(model, sampling_freq, num_folds, tca_type)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization=tca_type)
        
        #  combined shift
        shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
        f_in_tests = ['data/VNIR_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
        f_outs = ['results/transfer/VNIR_combined/combined_{}_{}_{}_{}_{}_{}'.format(f_t, f_s, model, sampling_freq, num_folds, tca_type) for (f_t, f_s) in shift_range]
        run_experiment(f_in_train='data/VNIR_combined/combined_0.0_0.0.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, standardization=tca_type)
        for f in f_outs:
            print(f)
            summarize_results([f], num_folds, f)
            

    # ------------------------
    # PDS
    # ------------------------
    pds_type = 'pds'
    #transversal
    shift_range = np.linspace(0,20,21)
    f_in_tests = ['data/SWIR3505_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
    f_outs = ['results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_{}_{}'.format(round(f_s), model, sampling_freq, num_folds, pds_type) for f_s in shift_range]
    f_in_train = 'data/SWIR3505_transversal/transversal_0.0.npz'
    f_out_summarized = 'results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_{}'.format(model, sampling_freq, num_folds, pds_type)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization=pds_type)


    # intensity
    shift_range = np.linspace(0,0.25,26)
    f_in_tests = ['data/SWIR3505_intensity_v2/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
    f_outs = ['results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_{}_{}'.format(round(f_s,2), model, sampling_freq, num_folds, pds_type) for f_s in shift_range]
    f_in_train = 'data/SWIR3505_intensity_v2/intensity_0.0.npz'
    f_out_summarized = 'results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_{}'.format(model, sampling_freq, num_folds, pds_type)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization=pds_type)

    #  combined shift

    shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
    f_in_tests = ['data/SWIR3505_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
    f_outs = ['results/transfer/SWIR3505_combined/combined_{}_{}_{}_{}_{}_{}'.format(f_t, f_s, model, sampling_freq, num_folds, pds_type) for (f_t, f_s) in shift_range]
    run_experiment(f_in_train='data/SWIR3505_combined/combined_0.0_0.0.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, standardization=pds_type)
    for f in f_outs:
        print(f)
        summarize_results([f], num_folds, f)

    sampling_freq=2
    pds_type = 'pds'
    # transversal
    shift_range = np.linspace(0,20,21)
    f_in_tests = ['data/VNIR_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
    f_outs = ['results/transfer/VNIR_transversal/transversal_{}_{}_{}_{}_{}'.format(round(f_s), model, sampling_freq, num_folds, pds_type) for f_s in shift_range]
    f_in_train = 'data/VNIR_transversal/transversal_0.0.npz'
    f_out_summarized = 'results/transfer/VNIR_transversal/transversal_{}_{}_{}_{}'.format(model, sampling_freq, num_folds, pds_type)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization=pds_type)


    # intensity
    shift_range = np.linspace(0,0.25,26)
    f_in_tests = ['data/VNIR_intensity/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
    f_outs = ['results/transfer/VNIR_intensity/intensity_{}_{}_{}_{}_{}'.format(round(f_s,2), model, sampling_freq, num_folds, pds_type) for f_s in shift_range]
    f_in_train = 'data/VNIR_intensity/intensity_0.0.npz'
    f_out_summarized = 'results/transfer/VNIR_intensity/intensity_{}_{}_{}_{}'.format(model, sampling_freq, num_folds, pds_type)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization=pds_type)

    #  combined shift

    shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
    f_in_tests = ['data/VNIR_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
    f_outs = ['results/transfer/VNIR_combined/combined_{}_{}_{}_{}_{}_{}'.format(f_t, f_s, model, sampling_freq, num_folds, pds_type) for (f_t, f_s) in shift_range]
    run_experiment(f_in_train='data/VNIR_combined/combined_0.0_0.0.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, standardization=pds_type)
    for f in f_outs:
        print(f)
        summarize_results([f], num_folds, f)

    # ------------------------
    # PDS + cheb
    # ------------------------

    for pds_type in ['pds']:
        #transversal
        shift_range = np.linspace(0,20,21)
        f_in_tests = ['data/SWIR3505_transversal/transversal_{}_cheb.npz'.format(round(f_s, 1)) for f_s in shift_range]
        f_outs = ['results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_{}_cheb_{}'.format(round(f_s), model, 1, num_folds, pds_type) for f_s in shift_range]
        f_in_train = 'data/SWIR3505_transversal/transversal_0.0_cheb.npz'
        f_out_summarized = 'results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_cheb_{}'.format(model, 1, num_folds, pds_type)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, 1, standardization=pds_type)


        # intensity
        shift_range = np.linspace(0,0.25,26)
        f_in_tests = ['data/SWIR3505_intensity_v2/intensity_{}_cheb.npz'.format(round(f_s, 2)) for f_s in shift_range]
        f_outs = ['results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_{}_cheb_{}'.format(round(f_s,2), model, 1, num_folds, pds_type) for f_s in shift_range]
        f_in_train = 'data/SWIR3505_intensity_v2/intensity_0.0_cheb.npz'
        f_out_summarized = 'results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_cheb_{}'.format(model, 1, num_folds, pds_type)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, 1, standardization=pds_type)

        #  combined shift
        shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
        f_in_tests = ['data/SWIR3505_combined/combined_{}_{}_cheb.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
        f_outs = ['results/transfer/SWIR3505_combined/combined_{}_{}_{}_{}_{}_cheb_{}'.format(f_t, f_s, model, 1, num_folds, pds_type) for (f_t, f_s) in shift_range]
        run_experiment(f_in_train='data/SWIR3505_combined/combined_0.0_0.0_cheb.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=1, standardization=pds_type)
        for f in f_outs:
            print(f)
            summarize_results([f], num_folds, f)


    # ------------------------------------------------------------------------------
    # Reversing transversal shift
    # ------------------------------------------------------------------------------
    
    # ------------------------
    # DTW
    # ------------------------

    #transversal
    shift_range = np.linspace(0,20,21)
    f_in_tests = ['data/SWIR3505_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
    f_outs = ['results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_{}_dtw'.format(round(f_s), model, sampling_freq, num_folds) for f_s in shift_range]
    f_in_train = 'data/SWIR3505_transversal/transversal_0.0.npz'
    f_out_summarized = 'results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_dtw'.format(model, sampling_freq, num_folds)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization='dtw')


    # intensity
    shift_range = np.linspace(0,0.25,26)
    f_in_tests = ['data/SWIR3505_intensity_v2/intensity_{}_cheb.npz'.format(round(f_s, 2)) for f_s in shift_range]
    f_outs = ['results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_{}_cheb_{}'.format(round(f_s,2), model, 1, num_folds, pds_type) for f_s in shift_range]
    f_in_train = 'data/SWIR3505_intensity_v2/intensity_0.0_cheb.npz'
    f_out_summarized = 'results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_cheb_{}'.format(model, 1, num_folds, pds_type)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, 1, standardization=pds_type)
    

    #  combined shift
    shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
    f_in_tests = ['data/SWIR3505_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
    f_outs = ['results/transfer/SWIR3505_combined/combined_{}_{}_{}_{}_{}_dtw'.format(f_t, f_s, model, sampling_freq, num_folds) for (f_t, f_s) in shift_range]
    run_experiment(f_in_train='data/SWIR3505_combined/combined_0.0_0.0.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, standardization='dtw')
    for f in f_outs:
        print(f)
        summarize_results([f], num_folds, f)


    # transversal
    sampling_freq = 18
    shift_range = np.linspace(0,20,21)
    f_in_tests = ['data/SWIR3505_transversal/transversal_{}_interpolated.npz'.format(round(f_s, 1)) for f_s in shift_range]
    f_outs = ['results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_{}_dtw-full'.format(round(f_s), model, sampling_freq, num_folds) for f_s in shift_range]
    f_in_train = 'data/SWIR3505_transversal/transversal_0.0_interpolated.npz'
    f_out_summarized = 'results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_dtw-full'.format(model, sampling_freq, num_folds)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization='dtw-full')
        

    # intensity
    shift_range = np.linspace(0,0.25,26)
    f_in_tests = ['data/SWIR3505_intensity_v2/intensity_{}_interpolated.npz'.format(round(f_s, 2)) for f_s in shift_range]
    f_outs = ['results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_{}_dtw_full'.format(round(f_s,2), model, sampling_freq, num_folds) for f_s in shift_range]
    f_in_train = 'data/SWIR3505_intensity_v2/intensity_0.0_interpolated.npz'
    f_out_summarized = 'results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_dtw_full'.format(model, sampling_freq, num_folds)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization='_dtw_full')
    

    #  combined shift
    shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
    f_in_tests = ['data/SWIR3505_combined/combined_{}_{}_interpolated.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
    f_outs = ['results/transfer/SWIR3505_combined/combined_{}_{}_{}_{}_{}_dtw-full'.format(f_t, f_s, model, sampling_freq, num_folds) for (f_t, f_s) in shift_range]
    run_experiment(f_in_train='data/SWIR3505_combined/combined_0.0_0.0_interpolated.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, standardization='dtw-full')
    for f in f_outs:
        print(f)
        summarize_results([f], num_folds, f)

    # #transversal
    sampling_freq = 7
    shift_range = np.linspace(0,20,21)
    f_in_tests = ['data/VNIR_transversal/transversal_{}_interpolated.npz'.format(round(f_s, 1)) for f_s in shift_range]
    f_outs = ['results/transfer/VNIR_transversal/transversal_{}_{}_{}_{}_dtw-full'.format(round(f_s), model, sampling_freq, num_folds) for f_s in shift_range]
    f_in_train = 'data/VNIR_transversal/transversal_0.0_interpolated.npz'
    f_out_summarized = 'results/transfer/VNIR_transversal/transversal_{}_{}_{}_dtw-full'.format(model, sampling_freq, num_folds)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization='dtw-full')
        

    # intensity
    shift_range = np.linspace(0,0.25,26)
    f_in_tests = ['data/VNIR_intensity/intensity_{}_interpolated.npz'.format(round(f_s, 2)) for f_s in shift_range]
    f_outs = ['results/transfer/VNIR_intensity/intensity_{}_{}_{}_{}_dtw_full'.format(round(f_s,2), model, sampling_freq, num_folds) for f_s in shift_range]
    f_in_train = 'data/VNIR_intensity/intensity_0.0_interpolated.npz'
    f_out_summarized = 'results/transfer/VNIR_intensity/intensity_{}_{}_{}_dtw_full'.format(model, sampling_freq, num_folds)
    perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization='_dtw_full')
    

    #  combined shift
    shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
    f_in_tests = ['data/VNIR_combined/combined_{}_{}_interpolated.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
    f_outs = ['results/transfer/VNIR_combined/combined_{}_{}_{}_{}_{}_dtw-full'.format(f_t, f_s, model, sampling_freq, num_folds) for (f_t, f_s) in shift_range]
    run_experiment(f_in_train='data/VNIR_combined/combined_0.0_0.0_interpolated.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, standardization='dtw-full')
    for f in f_outs:
        print(f)
        summarize_results([f], num_folds, f)


    sampling_freq = 3
            

    # ------------------------
    # Moment matching
    # ------------------------

    for matching_strategy in ['mm1', 'mm2', 'mm3', 'mm4']:

        #transversal
        shift_range = np.linspace(0,20,21)
        f_in_tests = ['data/SWIR3505_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
        f_outs = ['results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_{}_{}'.format(round(f_s), model, sampling_freq, num_folds, matching_strategy) for f_s in shift_range]
        f_in_train = 'data/SWIR3505_transversal/transversal_0.0.npz'
        f_out_summarized = 'results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_{}'.format(model, sampling_freq, num_folds, matching_strategy)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization=matching_strategy)
            

        # intensity
        shift_range = np.linspace(0,0.25,26)
        f_in_tests = ['data/SWIR3505_intensity_v2/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
        f_outs = ['results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_{}_{}'.format(round(f_s,2), model, sampling_freq, num_folds, matching_strategy) for f_s in shift_range]
        f_in_train = 'data/SWIR3505_intensity_v2/intensity_0.0.npz'
        f_out_summarized = 'results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_{}'.format(model, sampling_freq, num_folds, matching_strategy)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization=matching_strategy)
        

        #  combined shift
        shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
        f_in_tests = ['data/SWIR3505_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
        f_outs = ['results/transfer/SWIR3505_combined/combined_{}_{}_{}_{}_{}_{}'.format(f_t, f_s, model, sampling_freq, num_folds, matching_strategy) for (f_t, f_s) in shift_range]
        run_experiment(f_in_train='data/SWIR3505_combined/combined_0.0_0.0.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, standardization=matching_strategy)
        for f in f_outs:
            print(f)
            summarize_results([f], num_folds, f)

    sampling_freq = 2
    for matching_strategy in ['mm1', 'mm2', 'mm3', 'mm4']:

        #transversal
        shift_range = np.linspace(0,20,21)
        f_in_tests = ['data/VNIR_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
        f_outs = ['results/transfer/VNIR_transversal/transversal_{}_{}_{}_{}_{}'.format(round(f_s), model, sampling_freq, num_folds, matching_strategy) for f_s in shift_range]
        f_in_train = 'data/VNIR_transversal/transversal_0.0.npz'
        f_out_summarized = 'results/transfer/VNIR_transversal/transversal_{}_{}_{}_{}'.format(model, sampling_freq, num_folds, matching_strategy)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization=matching_strategy)
            

        # intensity
        shift_range = np.linspace(0,0.25,26)
        f_in_tests = ['data/VNIR_intensity/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
        f_outs = ['results/transfer/VNIR_intensity/intensity_{}_{}_{}_{}_{}'.format(round(f_s,2), model, sampling_freq, num_folds, matching_strategy) for f_s in shift_range]
        f_in_train = 'data/VNIR_intensity/intensity_0.0.npz'
        f_out_summarized = 'results/transfer/VNIR_intensity/intensity_{}_{}_{}_{}'.format(model, sampling_freq, num_folds, matching_strategy)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization=matching_strategy)
        

        #  combined shift
        shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
        f_in_tests = ['data/VNIR_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
        f_outs = ['results/transfer/VNIR_combined/combined_{}_{}_{}_{}_{}_{}'.format(f_t, f_s, model, sampling_freq, num_folds, matching_strategy) for (f_t, f_s) in shift_range]
        run_experiment(f_in_train='data/VNIR_combined/combined_0.0_0.0.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, standardization=matching_strategy)
        for f in f_outs:
            print(f)
            summarize_results([f], num_folds, f)



    # ------------------------------------------------------------------------------
    # Robust Training Strategies
    # ------------------------------------------------------------------------------
    
    # ------------------------
    # Robust training intensity
    # ------------------------

    for rob_shift in [0.025]:
        #transversal
        shift_range = np.linspace(0,20,21)
        f_in_tests = ['data/SWIR3505_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
        f_outs = ['results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_{}_rob_training_intensity_{}'.format(round(f_s), model, sampling_freq, num_folds, rob_shift) for f_s in shift_range]
        f_in_train = 'data/SWIR3505_transversal/transversal_0.0.npz'
        f_out_summarized = 'results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_rob_training_intensity_{}'.format(model, sampling_freq, num_folds, rob_shift)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_train_intensity=rob_shift)
        
        # intensity
        shift_range = np.linspace(0,0.25,26)
        f_in_tests = ['data/SWIR3505_intensity_v2/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
        f_outs = ['results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_{}_rob_training_intensity_{}'.format(round(f_s,2), model, sampling_freq, num_folds, rob_shift) for f_s in shift_range]
        f_in_train = 'data/SWIR3505_intensity_v2/intensity_0.0.npz'
        f_out_summarized = 'results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_rob_training_intensity_{}'.format(model, sampling_freq, num_folds, rob_shift)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_train_intensity=rob_shift)
    
        # combined 
        shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
        f_in_tests = ['data/SWIR3505_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
        f_outs = ['results/transfer/SWIR3505_combined/combined_{}_{}_{}_{}_{}_rob_training_intensity_{}'.format(f_t, f_s, model, sampling_freq, num_folds, rob_shift) for (f_t, f_s) in shift_range]
        run_experiment(f_in_train='data/SWIR3505_combined/combined_0.0_0.0.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, rob_train_intensity=rob_shift)
        for f in f_outs:
            summarize_results([f], num_folds, f)

    sampling_freq = 2
    for rob_shift in [0.025]:
        #transversal
        shift_range = np.linspace(0,20,21)
        f_in_tests = ['data/VNIR_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
        f_outs = ['results/transfer/VNIR_transversal/transversal_{}_{}_{}_{}_rob_training_intensity_{}'.format(round(f_s), model, sampling_freq, num_folds, rob_shift) for f_s in shift_range]
        f_in_train = 'data/VNIR_transversal/transversal_0.0.npz'
        f_out_summarized = 'results/transfer/VNIR_transversal/transversal_{}_{}_{}_rob_training_intensity_{}'.format(model, sampling_freq, num_folds, rob_shift)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_train_intensity=rob_shift)
        
        # intensity
        shift_range = np.linspace(0,0.25,26)
        f_in_tests = ['data/VNIR_intensity/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
        f_outs = ['results/transfer/VNIR_intensity/intensity_{}_{}_{}_{}_rob_training_intensity_{}'.format(round(f_s,2), model, sampling_freq, num_folds, rob_shift) for f_s in shift_range]
        f_in_train = 'data/VNIR_intensity/intensity_0.0.npz'
        f_out_summarized = 'results/transfer/VNIR_intensity/intensity_{}_{}_{}_rob_training_intensity_{}'.format(model, sampling_freq, num_folds, rob_shift)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_train_intensity=rob_shift)
    
        # combined 
        shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
        f_in_tests = ['data/VNIR_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
        f_outs = ['results/transfer/VNIR_combined/combined_{}_{}_{}_{}_{}_rob_training_intensity_{}'.format(f_t, f_s, model, sampling_freq, num_folds, rob_shift) for (f_t, f_s) in shift_range]
        run_experiment(f_in_train='data/VNIR_combined/combined_0.0_0.0.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, rob_train_intensity=rob_shift)
        for f in f_outs:
            summarize_results([f], num_folds, f)

    
    # ------------------------
    # Robust training transversal
    # ------------------------

    for rob_shift, shift_list in [(3, [-3,-2,-1,1,2,3]),(5, [-5,-4,-3,-2,-1,1,2,3,4,5])]:#

        # transversal
        shift_range = np.linspace(0,20,21)
        f_in_tests = ['data/SWIR3505_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
        f_outs = ['results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_{}_rob_training_transversal_{}'.format(round(f_s), model, sampling_freq, num_folds, rob_shift) for f_s in shift_range]
        f_in_train = 'data/SWIR3505_transversal/transversal_0.0.npz'
        f_out_summarized = 'results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_rob_training_transversal_{}'.format(model, sampling_freq, num_folds, rob_shift)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_train_transversal=('data/SWIR3505_transversal/transversal_for_robust_training', shift_list))

        # intensity
        shift_range = np.linspace(0,0.25,26)
        f_in_tests = ['data/SWIR3505_intensity_v2/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
        f_outs = ['results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_{}_rob_training_transversal_{}'.format(round(f_s,2), model, sampling_freq, num_folds, rob_shift) for f_s in shift_range]
        f_in_train = 'data/SWIR3505_intensity_v2/intensity_0.0.npz'
        f_out_summarized = 'results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_rob_training_transversal_{}'.format(model, sampling_freq, num_folds, rob_shift)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_train_transversal=('data/SWIR3505_transversal/transversal_for_robust_training', shift_list))
    
        # combined 
        shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
        f_in_tests = ['data/SWIR3505_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
        f_outs = ['results/transfer/SWIR3505_combined/combined_{}_{}_{}_{}_{}_rob_training_transversal_{}'.format(f_t, f_s, model, sampling_freq, num_folds, rob_shift) for (f_t, f_s) in shift_range]
        run_experiment(f_in_train='data/SWIR3505_combined/combined_0.0_0.0.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, rob_train_transversal=('data/SWIR3505_transversal/transversal_for_robust_training', shift_list))
        for f in f_outs:
            summarize_results([f], num_folds, f)
        
    sampling_freq = 2
    for rob_shift, shift_list in [(3, [-3,-2,-1,1,2,3]),(5, [-5,-4,-3,-2,-1,1,2,3,4,5])]:#

        # transversal
        shift_range = np.linspace(0,20,21)
        f_in_tests = ['data/VNIR_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
        f_outs = ['results/transfer/VNIR_transversal/transversal_{}_{}_{}_{}_rob_training_transversal_{}'.format(round(f_s), model, sampling_freq, num_folds, rob_shift) for f_s in shift_range]
        f_in_train = 'data/VNIR_transversal/transversal_0.0.npz'
        f_out_summarized = 'results/transfer/VNIR_transversal/transversal_{}_{}_{}_rob_training_transversal_{}'.format(model, sampling_freq, num_folds, rob_shift)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_train_transversal=('data/VNIR_transversal/transversal_for_robust_training', shift_list))

        # intensity
        shift_range = np.linspace(0,0.25,26)
        f_in_tests = ['data/VNIR_intensity/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
        f_outs = ['results/transfer/VNIR_intensity/intensity_{}_{}_{}_{}_rob_training_transversal_{}'.format(round(f_s,2), model, sampling_freq, num_folds, rob_shift) for f_s in shift_range]
        f_in_train = 'data/VNIR_intensity/intensity_0.0.npz'
        f_out_summarized = 'results/transfer/VNIR_intensity/intensity_{}_{}_{}_rob_training_transversal_{}'.format(model, sampling_freq, num_folds, rob_shift)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_train_transversal=('data/VNIR_transversal/transversal_for_robust_training', shift_list))
    
        # combined 
        shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
        f_in_tests = ['data/VNIR_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
        f_outs = ['results/transfer/VNIR_combined/combined_{}_{}_{}_{}_{}_rob_training_transversal_{}'.format(f_t, f_s, model, sampling_freq, num_folds, rob_shift) for (f_t, f_s) in shift_range]
        run_experiment(f_in_train='data/VNIR_combined/combined_0.0_0.0.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, rob_train_transversal=('data/VNIR_transversal/transversal_for_robust_training', shift_list))
        for f in f_outs:
            summarize_results([f], num_folds, f)

    # # ------------------------
    # # Robust feature selection 
    # # ------------------------
    for eta in [1.2]:#

        # transversal
        shift_range = np.linspace(0,20,21)
        f_in_tests = ['data/SWIR3505_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
        f_outs = ['results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_{}_rob_features_{}'.format(round(f_s), model, sampling_freq, num_folds, eta) for f_s in shift_range]
        f_in_train = 'data/SWIR3505_transversal/transversal_0.0.npz'
        f_out_summarized = 'results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_rob_features_{}'.format(model, sampling_freq, num_folds, eta)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_features='data/SWIR3505_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(eta, 1))

        # intensity
        shift_range = np.linspace(0,0.25,26)
        f_in_tests = ['data/SWIR3505_intensity_v2/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
        f_outs = ['results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_{}_rob_features_{}'.format(round(f_s,2), model, sampling_freq, num_folds, eta) for f_s in shift_range]
        f_in_train = 'data/SWIR3505_intensity_v2/intensity_0.0.npz'
        f_out_summarized = 'results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_rob_features_{}'.format(model, sampling_freq, num_folds, eta)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_features='data/SWIR3505_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(eta, 1))
    
        # combined 
        shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
        f_in_tests = ['data/SWIR3505_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
        f_outs = ['results/transfer/SWIR3505_combined/combined_{}_{}_{}_{}_{}_rob_features_{}'.format(f_t, f_s, model, sampling_freq, num_folds, eta) for (f_t, f_s) in shift_range]
        run_experiment(f_in_train='data/SWIR3505_combined/combined_0.0_0.0.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, rob_features='data/SWIR3505_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(eta, 1))
        for f in f_outs:
            summarize_results([f], num_folds, f)

    sampling_freq = 2
    for eta in [1.2]:#

        # transversal
        shift_range = np.linspace(0,20,21)
        f_in_tests = ['data/VNIR_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
        f_outs = ['results/transfer/VNIR_transversal/transversal_{}_{}_{}_{}_rob_features_{}'.format(round(f_s), model, sampling_freq, num_folds, eta) for f_s in shift_range]
        f_in_train = 'data/VNIR_transversal/transversal_0.0.npz'
        f_out_summarized = 'results/transfer/VNIR_transversal/transversal_{}_{}_{}_rob_features_{}'.format(model, sampling_freq, num_folds, eta)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_features='data/VNIR_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(eta, 1))

        # intensity
        shift_range = np.linspace(0,0.25,26)
        f_in_tests = ['data/VNIR_intensity/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
        f_outs = ['results/transfer/VNIR_intensity/intensity_{}_{}_{}_{}_rob_features_{}'.format(round(f_s,2), model, sampling_freq, num_folds, eta) for f_s in shift_range]
        f_in_train = 'data/VNIR_intensity/intensity_0.0.npz'
        f_out_summarized = 'results/transfer/VNIR_intensity/intensity_{}_{}_{}_rob_features_{}'.format(model, sampling_freq, num_folds, eta)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_features='data/VNIR_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(eta, 1))
    
        # combined 
        shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
        f_in_tests = ['data/VNIR_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
        f_outs = ['results/transfer/VNIR_combined/combined_{}_{}_{}_{}_{}_rob_features_{}'.format(f_t, f_s, model, sampling_freq, num_folds, eta) for (f_t, f_s) in shift_range]
        run_experiment(f_in_train='data/VNIR_combined/combined_0.0_0.0.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, rob_features='data/VNIR_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(eta, 1))
        for f in f_outs:
            summarize_results([f], num_folds, f)

    # ------------------------
    # Robust feature selection + robust training
    # ------------------------

    for eta in [1.2]:
        for rob_shift, shift_list in [(3, [-3,-2,-1,1,2,3])]:

            # transversal
            shift_range = np.linspace(0,20,21)
            f_in_tests = ['data/SWIR3505_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
            f_outs = ['results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_{}_rob_features_{}_rob_training_transversal_{}'.format(round(f_s), model, sampling_freq, num_folds, eta, rob_shift) for f_s in shift_range]
            f_in_train = 'data/SWIR3505_transversal/transversal_0.0.npz'
            f_out_summarized = 'results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_rob_features_{}_rob_training_transversal_{}'.format(model, sampling_freq, num_folds, eta, rob_shift)
            perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_features='data/SWIR3505_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(eta, 1), rob_train_transversal=('data/SWIR3505_transversal/transversal_for_robust_training', shift_list))

            # intensity
            shift_range = np.linspace(0,0.25,26)
            f_in_tests = ['data/SWIR3505_intensity_v2/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
            f_outs = ['results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_{}_rob_features_{}_rob_training_transversal_{}'.format(round(f_s,2), model, sampling_freq, num_folds, eta, rob_shift) for f_s in shift_range]
            f_in_train = 'data/SWIR3505_intensity_v2/intensity_0.0.npz'
            f_out_summarized = 'results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_rob_features_{}_rob_training_transversal_{}'.format(model, sampling_freq, num_folds, eta, rob_shift)
            perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_features='data/SWIR3505_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(eta, 1), rob_train_transversal=('data/SWIR3505_transversal/transversal_for_robust_training', shift_list))
        
            # combined 
            shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
            f_in_tests = ['data/SWIR3505_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
            f_outs = ['results/transfer/SWIR3505_combined/combined_{}_{}_{}_{}_{}_rob_features_{}_rob_training_transversal_{}'.format(f_t, f_s, model, sampling_freq, num_folds, eta, rob_shift) for (f_t, f_s) in shift_range]
            run_experiment(f_in_train='data/SWIR3505_combined/combined_0.0_0.0.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, rob_features='data/SWIR3505_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(eta, 1), rob_train_transversal=('data/SWIR3505_transversal/transversal_for_robust_training', shift_list))
            for f in f_outs:
                summarize_results([f], num_folds, f)

    sampling_freq = 2
    for eta in [1.2]:
        for rob_shift, shift_list in [(3, [-3,-2,-1,1,2,3])]:

            # transversal
            shift_range = np.linspace(0,20,21)
            f_in_tests = ['data/VNIR_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
            f_outs = ['results/transfer/VNIR_transversal/transversal_{}_{}_{}_{}_rob_features_{}_rob_training_transversal_{}'.format(round(f_s), model, sampling_freq, num_folds, eta, rob_shift) for f_s in shift_range]
            f_in_train = 'data/VNIR_transversal/transversal_0.0.npz'
            f_out_summarized = 'results/transfer/VNIR_transversal/transversal_{}_{}_{}_rob_features_{}_rob_training_transversal_{}'.format(model, sampling_freq, num_folds, eta, rob_shift)
            perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_features='data/VNIR_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(eta, 1), rob_train_transversal=('data/VNIR_transversal/transversal_for_robust_training', shift_list))

            # intensity
            shift_range = np.linspace(0,0.25,26)
            f_in_tests = ['data/VNIR_intensity/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
            f_outs = ['results/transfer/VNIR_intensity/intensity_{}_{}_{}_{}_rob_features_{}_rob_training_transversal_{}'.format(round(f_s,2), model, sampling_freq, num_folds, eta, rob_shift) for f_s in shift_range]
            f_in_train = 'data/VNIR_intensity/intensity_0.0.npz'
            f_out_summarized = 'results/transfer/VNIR_intensity/intensity_{}_{}_{}_rob_features_{}_rob_training_transversal_{}'.format(model, sampling_freq, num_folds, eta, rob_shift)
            perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_features='data/VNIR_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(eta, 1), rob_train_transversal=('data/VNIR_transversal/transversal_for_robust_training', shift_list))
        
            # combined 
            shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
            f_in_tests = ['data/VNIR_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
            f_outs = ['results/transfer/VNIR_combined/combined_{}_{}_{}_{}_{}_rob_features_{}_rob_training_transversal_{}'.format(f_t, f_s, model, sampling_freq, num_folds, eta, rob_shift) for (f_t, f_s) in shift_range]
            run_experiment(f_in_train='data/VNIR_combined/combined_0.0_0.0.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, rob_features='data/VNIR_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(eta, 1), rob_train_transversal=('data/VNIR_transversal/transversal_for_robust_training', shift_list))
            for f in f_outs:
                summarize_results([f], num_folds, f)

    # ------------------------
    # Robust feature selection + robust training
    # ------------------------

    for eta in [1.2]:
        for rob_shift in [0.025]:

            # transversal
            shift_range = np.linspace(0,20,21)
            f_in_tests = ['data/SWIR3505_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
            f_outs = ['results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_{}_rob_features_{}_rob_training_intensity_{}'.format(round(f_s), model, sampling_freq, num_folds, eta, rob_shift) for f_s in shift_range]
            f_in_train = 'data/SWIR3505_transversal/transversal_0.0.npz'
            f_out_summarized = 'results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_rob_features_{}_rob_training_intensity_{}'.format(model, sampling_freq, num_folds, eta, rob_shift)
            perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_features='data/SWIR3505_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(eta, 1), rob_train_intensity=rob_shift)

            # intensity
            shift_range = np.linspace(0,0.25,26)
            f_in_tests = ['data/SWIR3505_intensity_v2/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
            f_outs = ['results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_{}_rob_features_{}_rob_training_rob_training_intensity_{}'.format(round(f_s,2), model, sampling_freq, num_folds, eta, rob_shift) for f_s in shift_range]
            f_in_train = 'data/SWIR3505_intensity_v2/intensity_0.0.npz'
            f_out_summarized = 'results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_rob_features_{}_rob_training_intensity_{}'.format(model, sampling_freq, num_folds, eta, rob_shift)
            perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_features='data/SWIR3505_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(eta, 1), rob_train_intensity=rob_shift)
        
            # combined 
            shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
            f_in_tests = ['data/SWIR3505_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
            f_outs = ['results/transfer/SWIR3505_combined/combined_{}_{}_{}_{}_{}_rob_features_{}_rob_training_intensity_{}'.format(f_t, f_s, model, sampling_freq, num_folds, eta, rob_shift) for (f_t, f_s) in shift_range]
            run_experiment(f_in_train='data/SWIR3505_combined/combined_0.0_0.0.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, rob_features='data/SWIR3505_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(eta, 1), rob_train_intensity=rob_shift)
            for f in f_outs:
                summarize_results([f], num_folds, f)


    for eta in [1.2]:
        for rob_shift in [0.025]:

            # transversal
            shift_range = np.linspace(0,20,21)
            f_in_tests = ['data/VNIR_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
            f_outs = ['results/transfer/VNIR_transversal/transversal_{}_{}_{}_{}_rob_features_{}_rob_training_intensity_{}'.format(round(f_s), model, sampling_freq, num_folds, eta, rob_shift) for f_s in shift_range]
            f_in_train = 'data/VNIR_transversal/transversal_0.0.npz'
            f_out_summarized = 'results/transfer/VNIR_transversal/transversal_{}_{}_{}_rob_features_{}_rob_training_intensity_{}'.format(model, sampling_freq, num_folds, eta, rob_shift)
            perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_features='data/VNIR_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(eta, 1), rob_train_intensity=rob_shift)

            # intensity
            shift_range = np.linspace(0,0.25,26)
            f_in_tests = ['data/VNIR_intensity/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
            f_outs = ['results/transfer/VNIR_intensity/intensity_{}_{}_{}_{}_rob_features_{}_rob_training_rob_training_intensity_{}'.format(round(f_s,2), model, sampling_freq, num_folds, eta, rob_shift) for f_s in shift_range]
            f_in_train = 'data/VNIR_intensity/intensity_0.0.npz'
            f_out_summarized = 'results/transfer/VNIR_intensity/intensity_{}_{}_{}_rob_features_{}_rob_training_intensity_{}'.format(model, sampling_freq, num_folds, eta, rob_shift)
            perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_features='data/VNIR_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(eta, 1), rob_train_intensity=rob_shift)
        
            # combined 
            shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
            f_in_tests = ['data/VNIR_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
            f_outs = ['results/transfer/VNIR_combined/combined_{}_{}_{}_{}_{}_rob_features_{}_rob_training_intensity_{}'.format(f_t, f_s, model, sampling_freq, num_folds, eta, rob_shift) for (f_t, f_s) in shift_range]
            run_experiment(f_in_train='data/VNIR_combined/combined_0.0_0.0.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, rob_features='data/VNIR_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(eta, 1), rob_train_intensity=rob_shift)
            for f in f_outs:
                summarize_results([f], num_folds, f)

    # ------------------------
    # offset elimination with limited standardization samples
    # ------------------------

    for num_samples in [10, 20, 30, 40, 50, 100, 200, 500]:

        #transversal
        shift_range = np.linspace(0,20,21)
        f_in_tests = ['data/SWIR3505_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
        f_outs = ['results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_{}_offset_elim_{}'.format(round(f_s), model, sampling_freq, num_folds, num_samples) for f_s in shift_range]
        f_in_train = 'data/SWIR3505_transversal/transversal_0.0.npz'
        f_out_summarized ='results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_offset_elim_{}'.format(model, sampling_freq, num_folds, num_samples)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization='mean', standardization_samples=num_samples)
        
        # intensity
        shift_range = [0.25]# np.linspace(0,0.25,26)
        f_in_tests = ['data/SWIR3505_intensity_v2/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
        f_outs = ['results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_{}_offset_elim_{}'.format(round(f_s,2), model, sampling_freq, num_folds, num_samples) for f_s in shift_range]
        f_in_train = 'data/SWIR3505_intensity_v2/intensity_0.0.npz'
        f_out_summarized = 'results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_offset_elim_{}'.format(model, sampling_freq, num_folds, num_samples)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization='mean', standardization_samples=num_samples)

        #  combined shift
        shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
        f_in_tests = ['data/SWIR3505_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
        f_outs = ['results/transfer/SWIR3505_combined/combined_{}_{}_{}_{}_{}_offset_elim_{}'.format(f_t, f_s, model, sampling_freq, num_folds, num_samples) for (f_t, f_s) in shift_range]
        run_experiment(f_in_train='data/SWIR3505_combined/combined_0.0_0.0.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, standardization='mean', standardization_samples=num_samples)
        for f in f_outs:
            print(f)
            summarize_results([f], num_folds, f)

    # # ------------------------
    # # offset elimination with limited standardization samples
    # # ------------------------

    for prepr_class in [0, 1, 2]:

        #transversal
        shift_range = np.linspace(0,20,21)
        f_in_tests = ['data/SWIR3505_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
        f_outs = ['results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_{}_offset_elim_c{}'.format(round(f_s), model, sampling_freq, num_folds, prepr_class) for f_s in shift_range]
        f_in_train = 'data/SWIR3505_transversal/transversal_0.0.npz'
        f_out_summarized ='results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_offset_elim_c{}'.format(model, sampling_freq, num_folds, prepr_class)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization='mean', prepr_class=prepr_class)
        
        # intensity
        shift_range = [0.25]# np.linspace(0,0.25,26)
        f_in_tests = ['data/SWIR3505_intensity_v2/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
        f_outs = ['results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_{}_offset_elim_c{}'.format(round(f_s,2), model, sampling_freq, num_folds, prepr_class) for f_s in shift_range]
        f_in_train = 'data/SWIR3505_intensity_v2/intensity_0.0.npz'
        f_out_summarized = 'results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_offset_elim_c{}'.format(model, sampling_freq, num_folds, prepr_class)
        perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, standardization='mean', prepr_class=prepr_class)

        #  combined shift
        shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
        f_in_tests = ['data/SWIR3505_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
        f_outs = ['results/transfer/SWIR3505_combined/combined_{}_{}_{}_{}_{}_offset_elim_c{}'.format(f_t, f_s, model, sampling_freq, num_folds, prepr_class) for (f_t, f_s) in shift_range]
        run_experiment(f_in_train='data/SWIR3505_combined/combined_0.0_0.0.npz', f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, standardization='mean', prepr_class=prepr_class)
        for f in f_outs:
            print(f)
            summarize_results([f], num_folds, f)


    # ------------------------
    # Robust feature selection with data standardization
    # ------------------------
    for strategy in ['dtw-full']:

        for eta in [1.2]:

            if strategy == 'dtw-full':
                sampling_freq = 18
            else:
                sampling_freq = 3
            # transversal
            shift_range = np.linspace(0,20,21)
            f_in_tests = ['data/SWIR3505_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
            f_outs = ['results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_{}_rob_features_{}_{}'.format(round(f_s), model, sampling_freq, num_folds, eta, strategy) for f_s in shift_range]
            f_in_train = 'data/SWIR3505_transversal/transversal_0.0.npz'
            f_out_summarized = 'results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_rob_features_{}_{}'.format(model, sampling_freq, num_folds, eta, strategy)
            if strategy == 'dtw-full':
                f_in_train = 'data/SWIR3505_transversal/transversal_0.0_interpolated.npz'
                f_in_tests = ['data/SWIR3505_transversal/transversal_{}_interpolated.npz'.format(round(f_s, 1)) for f_s in shift_range]
            perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_features='data/SWIR3505_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(eta, 1), standardization=strategy)

            # intensity
            if strategy != 'dtw-full':
                shift_range = np.linspace(0,0.25,26)
                f_in_tests = ['data/SWIR3505_intensity_v2/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
                f_outs = ['results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_{}_rob_features_{}_{}'.format(round(f_s,2), model, sampling_freq, num_folds, eta, strategy) for f_s in shift_range]
                f_in_train = 'data/SWIR3505_intensity_v2/intensity_0.0.npz'
                f_out_summarized = 'results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_rob_features_{}_{}'.format(model, sampling_freq, num_folds, eta, strategy)
                perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_features='data/SWIR3505_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(eta, 1), standardization=strategy)
        
            # combined 
            shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
            f_in_tests = ['data/SWIR3505_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
            f_outs = ['results/transfer/SWIR3505_combined/combined_{}_{}_{}_{}_{}_rob_features_{}_{}'.format(f_t, f_s, model, sampling_freq, num_folds, eta, strategy) for (f_t, f_s) in shift_range]
            f_in_train='data/SWIR3505_combined/combined_0.0_0.0.npz'
            if strategy == 'dtw-full':
                f_in_train = 'data/SWIR3505_combined/combined_0.0_0.0_interpolated.npz'
                f_in_tests = ['data/SWIR3505_combined/combined_{}_interpolated.npz'.format(round(f_s, 1)) for f_s in shift_range]
            run_experiment(f_in_train=f_in_train, f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, rob_features='data/SWIR3505_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(eta, 1), standardization=strategy)
            for f in f_outs:
                summarize_results([f], num_folds, f)

    for strategy in  ['mean','dtw-full', 'mm1', 'mm2', 'mm3', 'mm4']:

        if strategy == 'dtw-full':
            sampling_freq = 7
        else:
            sampling_freq = 2

        for eta in [1.2]:

            # transversal
            shift_range = np.linspace(0,20,21)
            f_in_tests = ['data/VNIR_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
            f_outs = ['results/transfer/VNIR_transversal/transversal_{}_{}_{}_{}_rob_features_{}_{}'.format(round(f_s), model, sampling_freq, num_folds, eta, strategy) for f_s in shift_range]
            f_in_train = 'data/VNIR_transversal/transversal_0.0.npz'
            f_out_summarized = 'results/transfer/VNIR_transversal/transversal_{}_{}_{}_rob_features_{}_{}'.format(model, sampling_freq, num_folds, eta, strategy)
            if strategy == 'dtw-full':
                f_in_train = 'data/VNIR_transversal/transversal_0.0_interpolated.npz'
                f_in_tests = ['data/VNIR_transversal/transversal_{}_interpolated.npz'.format(round(f_s, 1)) for f_s in shift_range]
            perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_features='data/VNIR_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(eta, 1), standardization=strategy)

            if strategy != 'dtw-full':
                # intensity
                shift_range = np.linspace(0,0.25,26)
                f_in_tests = ['data/VNIR_intensity/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
                f_outs = ['results/transfer/VNIR_intensity/intensity_{}_{}_{}_{}_rob_features_{}_{}'.format(round(f_s,2), model, sampling_freq, num_folds, eta, strategy) for f_s in shift_range]
                f_in_train = 'data/VNIR_intensity/intensity_0.0.npz'
                f_out_summarized = 'results/transfer/VNIR_intensity/intensity_{}_{}_{}_rob_features_{}_{}'.format(model, sampling_freq, num_folds, eta, strategy)
                perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_features='data/VNIR_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(eta, 1), standardization=strategy)
        
            # combined 
            shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
            f_in_tests = ['data/VNIR_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
            f_outs = ['results/transfer/VNIR_combined/combined_{}_{}_{}_{}_{}_rob_features_{}_{}'.format(f_t, f_s, model, sampling_freq, num_folds, eta, strategy) for (f_t, f_s) in shift_range]
            f_in_train = 'data/VNIR_combined/combined_0.0_0.0.npz'
            if strategy == 'dtw-full':
                f_in_train = 'data/VNIR_combined/combined_0.0_0.0_interpolated.npz'
                f_in_tests = ['data/VNIR_combined/combined_{}_{}_interpolated.npz'.format(f_t, f_s) for (f_t,f_s) in shift_range]
            run_experiment(f_in_train=f_in_train, f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, rob_features='data/VNIR_transversal/robust_wavelengths_v2_{}_eta_{}.npz'.format(eta, 1), standardization=strategy)
            for f in f_outs:
                summarize_results([f], num_folds, f)

    
    # # # ------------------------
    # # # Robust training transversal
    # # # ------------------------
    for strategy in ['dtw-full']:
        for rob_shift, shift_list in [(3, [-3,-2,-1,1,2,3])]:#
            if strategy == 'dtw-full':
                sampling_freq = 18
            else:
                sampling_freq = 3

            # transversal
            shift_range = np.linspace(0,20,21)
            f_in_tests = ['data/SWIR3505_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
            f_outs = ['results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_{}_rob_training_transversal_{}_{}'.format(round(f_s), model, sampling_freq, num_folds, rob_shift,strategy) for f_s in shift_range]
            f_in_train = 'data/SWIR3505_transversal/transversal_0.0.npz'
            f_out_summarized = 'results/transfer/SWIR3505_transversal/transversal_{}_{}_{}_rob_training_transversal_{}_{}'.format(model, sampling_freq, num_folds, rob_shift,strategy)
            if strategy == 'dtw-full':
                f_in_train = 'data/SWIR3505_transversal/transversal_0.0_interpolated.npz'
                f_in_tests = ['data/SWIR3505_transversal/transversal_{}_interpolated.npz'.format(round(f_s, 1)) for f_s in shift_range]
            perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_train_transversal=('data/SWIR3505_transversal/transversal_for_robust_training', shift_list), standardization=strategy)

            if strategy != 'dtw-full':
                # intensity
                shift_range = np.linspace(0,0.25,26)
                f_in_tests = ['data/SWIR3505_intensity_v2/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
                f_outs = ['results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_{}_rob_training_transversal_{}_{}'.format(round(f_s,2), model, sampling_freq, num_folds, rob_shift,strategy) for f_s in shift_range]
                f_in_train = 'data/SWIR3505_intensity_v2/intensity_0.0.npz'
                f_out_summarized = 'results/transfer/SWIR3505_intensity_v2/intensity_{}_{}_{}_rob_training_transversal_{}_{}'.format(model, sampling_freq, num_folds, rob_shift,strategy)
                perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_train_transversal=('data/SWIR3505_transversal/transversal_for_robust_training', shift_list), standardization=strategy)
        
            # combined 
            shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
            f_in_tests = ['data/SWIR3505_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
            f_outs = ['results/transfer/SWIR3505_combined/combined_{}_{}_{}_{}_{}_rob_training_transversal_{}_{}'.format(f_t, f_s, model, sampling_freq, num_folds, rob_shift,strategy) for (f_t, f_s) in shift_range]
            f_in_train = 'data/SWIR3505_combined/combined_0.0_0.0.npz'
            if strategy == 'dtw-full':
                f_in_train = 'data/SWIR3505_combined/combined_0.0_0.0_interpolated.npz'
                f_in_tests = ['data/SWIR3505_combined/combined_{}_{}_interpolated.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
            run_experiment(f_in_train=f_in_train, f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, rob_train_transversal=('data/SWIR3505_transversal/transversal_for_robust_training', shift_list), standardization=strategy)
            for f in f_outs:
                summarize_results([f], num_folds, f)

    for strategy in ['dtw-full']:
        for rob_shift, shift_list in [(3, [-3,-2,-1,1,2,3])]:#
            if strategy == 'dtw-full':
                sampling_freq = 7
            else:
                sampling_freq = 2

            # transversal
            shift_range = np.linspace(0,20,21)
            f_in_tests = ['data/VNIR_transversal/transversal_{}.npz'.format(round(f_s, 1)) for f_s in shift_range]
            f_outs = ['results/transfer/VNIR_transversal/transversal_{}_{}_{}_{}_rob_training_transversal_{}_{}'.format(round(f_s), model, sampling_freq, num_folds, rob_shift,strategy) for f_s in shift_range]
            f_in_train = 'data/VNIR_transversal/transversal_0.0.npz'
            f_out_summarized = 'results/transfer/VNIR_transversal/transversal_{}_{}_{}_rob_training_transversal_{}_{}'.format(model, sampling_freq, num_folds, rob_shift,strategy)
            if strategy == 'dtw-full':
                f_in_train = 'data/VNIR_transversal/transversal_0.0_interpolated.npz'
                f_in_tests = ['data/VNIR_transversal/transversal_{}_interpolated.npz'.format(round(f_s, 1)) for f_s in shift_range]
            perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_train_transversal=('data/VNIR_transversal/transversal_for_robust_training', shift_list), standardization=strategy)

            if strategy != 'dtw-full':
                # intensity
                shift_range = np.linspace(0,0.25,26)
                f_in_tests = ['data/VNIR_intensity/intensity_{}.npz'.format(round(f_s, 2)) for f_s in shift_range]
                f_outs = ['results/transfer/VNIR_intensity/intensity_{}_{}_{}_{}_rob_training_transversal_{}_{}'.format(round(f_s,2), model, sampling_freq, num_folds, rob_shift,strategy) for f_s in shift_range]
                f_in_train = 'data/VNIR_intensity/intensity_0.0.npz'
                f_out_summarized = 'results/transfer/VNIR_intensity/intensity_{}_{}_{}_rob_training_transversal_{}_{}'.format(model, sampling_freq, num_folds, rob_shift,strategy)
                perform_experiment(shift_range, f_in_train, f_in_tests, f_outs, f_out_summarized, model, num_folds, sampling_freq, rob_train_transversal=('data/VNIR_transversal/transversal_for_robust_training', shift_list), standardization=strategy)
        
            # combined 
            shift_range = [(round(f_t, 1), round(f_s, 2)) for f_t in np.linspace(0,20,11) for f_s in np.linspace(0,0.25,6)]
            f_in_tests = ['data/VNIR_combined/combined_{}_{}.npz'.format(f_t, f_s) for (f_t, f_s) in shift_range]
            f_outs = ['results/transfer/VNIR_combined/combined_{}_{}_{}_{}_{}_rob_training_transversal_{}_{}'.format(f_t, f_s, model, sampling_freq, num_folds, rob_shift,strategy) for (f_t, f_s) in shift_range]
            f_in_train='data/VNIR_combined/combined_0.0_0.0.npz'
            if strategy == 'dtw-full':
                f_in_train = 'data/VNIR_combined/combined_0.0_0.0_interpolated.npz'
                f_in_tests = ['data/VNIR_combined/combined_{}_{}_interpolated.npz'.format(f_t,f_s) for f_t, f_s in shift_range]
            run_experiment(f_in_train=f_in_train, f_in_tests=f_in_tests, f_outs=f_outs, model=model, num_folds=num_folds, sampling_freq=sampling_freq, rob_train_transversal=('data/VNIR_transversal/transversal_for_robust_training', shift_list), standardization=strategy)
            for f in f_outs:
                summarize_results([f], num_folds, f)