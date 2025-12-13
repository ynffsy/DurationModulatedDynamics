"""Baseline inference by forecasting neural activity with rolling averages."""

import os
import time
import ipdb
import tqdm
import pickle
import numpy as np

from sklearn.model_selection import KFold

import scripts.config as config
import utils.utils_processing as utils_processing
import utils.utils_decoding as utils_decoding
import utils.utils_inference as utils_inference



# -----------------------------------------------------------------------------
# Shared configuration for the lightweight baseline inference benchmark
# -----------------------------------------------------------------------------
overwrite_results  = config.overwrite_results
data_dir           = config.data_dir
results_dir        = config.results_dir
session_data_names = config.session_data_names

unit_filters  = config.unit_filters
window_config = config.window_config
trial_filters = config.trial_filters
random_states = config.random_states
n_folds       = config.n_folds

data_formats  = config.data_formats
window_sizes  = config.window_sizes
    


def main(
    session_data_name,
    trial_filters,
    data_format):

    session_results_dir = os.path.join(results_dir, session_data_name)

    ## Initialize save name so each data format/version ends up in its own npz
    res_save_name = '_'.join(map(str, [x for x in [
        'inference_baseline',
        unit_filter,
        data_format] if x is not None]))
    
    res_save_path = os.path.join(session_results_dir, res_save_name + '.npz')
    if os.path.exists(res_save_path) and not overwrite_results:
        print('Results already exist for session: ', session_data_name)
        return

    ## Load spike trains (optionally resampled/padded)
    data_loader = utils_processing.DataLoaderDuo(
        data_dir,
        results_dir,
        session_data_name,
        unit_filter,
        None,
        window_config,
        trial_filters)
    
    data_loader.load_firing_rate_data()
    
    (firing_rates_slow_simple,       firing_rates_fast_simple, 
     input_firing_rates_slow_simple, input_firing_rates_fast_simple,
     trial_ids_slow,                 trial_ids_fast, 
     n_trials_slow,                  n_trials_fast,
     trial_lengths_slow,             trial_lengths_fast,
     times_new_slow,                 times_new_fast) = data_loader.reformat_firing_rate_data(data_format)

    ## Initialize results (train/test axis first to avoid confusion later)
    # GT_slow            = None
    # GT_fast            = None
    # baseline_slow_test = None
    # baseline_fast_test = None

    # if data_format in ['fill0', 'truncate_front', 'truncate_end', 'resample']:
    #     _, n_times_slow, n_neurons = firing_rates_slow_simple.shape
    #     _, n_times_fast, _         = firing_rates_fast_simple.shape

    #     GT_slow                 = firing_rates_slow_simple
    #     GT_fast                 = firing_rates_fast_simple
    #     baseline_slow_test      = np.zeros((len(random_states), len(window_sizes), n_trials_slow, n_times_slow, n_neurons))
    #     baseline_fast_test      = np.zeros((len(random_states), len(window_sizes), n_trials_slow, n_times_fast, n_neurons))
    
    forecast_slow_train = np.empty((len(random_states), len(window_sizes), n_trials_slow), dtype=object)
    forecast_slow_test  = np.empty((len(random_states), len(window_sizes), n_trials_slow), dtype=object)
    forecast_fast_train = np.empty((len(random_states), len(window_sizes), n_trials_fast), dtype=object)
    forecast_fast_test  = np.empty((len(random_states), len(window_sizes), n_trials_fast), dtype=object)

    ## The 1st dimension has 2 elements: 
    ##   baseline train, 
    ##   baseline test
    rmse_forecast_baseline_slow = np.zeros((2, len(random_states), n_folds, len(window_sizes)))
    rmse_forecast_baseline_fast = np.zeros((2, len(random_states), n_folds, len(window_sizes)))
    r2_forecast_baseline_slow   = np.zeros((2, len(random_states), n_folds, len(window_sizes)))
    r2_forecast_baseline_fast   = np.zeros((2, len(random_states), n_folds, len(window_sizes)))

    
    ## Read data and decode cursor states
    for i_rs, random_state in enumerate(random_states):

        ## Multi-fold cross validation
        np.random.seed(random_state)

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('random state: ', random_state)

        ## K-fold cross validation reuses the same folds for slow/fast activity
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        splits_slow = list(kf.split(np.arange(n_trials_slow)))
        splits_fast = list(kf.split(np.arange(n_trials_fast)))

        for i_fold, (
            (trial_indices_slow_train, trial_indices_slow_test), 
            (trial_indices_fast_train, trial_indices_fast_test)) in enumerate(zip(splits_slow, splits_fast)):
            
            print('fold: ', i_fold)

            if data_format is None:
                X_slow_train = [firing_rates_slow_simple[i] for i in trial_indices_slow_train]
                X_slow_test  = [firing_rates_slow_simple[i] for i in trial_indices_slow_test]
                X_fast_train = [firing_rates_fast_simple[i] for i in trial_indices_fast_train]
                X_fast_test  = [firing_rates_fast_simple[i] for i in trial_indices_fast_test]
            
            else:
                X_slow_train = np.array(firing_rates_slow_simple[trial_indices_slow_train])
                X_slow_test  = np.array(firing_rates_slow_simple[trial_indices_slow_test])
                X_fast_train = np.array(firing_rates_fast_simple[trial_indices_fast_train])
                X_fast_test  = np.array(firing_rates_fast_simple[trial_indices_fast_test])

            X_slow_train_concat = np.concatenate(X_slow_train, axis=0)
            X_slow_test_concat  = np.concatenate(X_slow_test,  axis=0)
            X_fast_train_concat = np.concatenate(X_fast_train, axis=0)
            X_fast_test_concat  = np.concatenate(X_fast_test,  axis=0)

            X_slow_train_trial_lengths = trial_lengths_slow[trial_indices_slow_train]
            X_slow_test_trial_lengths  = trial_lengths_slow[trial_indices_slow_test]
            X_fast_train_trial_lengths = trial_lengths_fast[trial_indices_fast_train]
            X_fast_test_trial_lengths  = trial_lengths_fast[trial_indices_fast_test]

            ## For baseline comparison, use the average of previous n timestamps 
            ##   as the prediction for the next timestamp. n should be varied
            for i_window, window_size in enumerate(window_sizes):
                
                ## Baseline on train data
                X_forecast_slow_train_baseline = utils_inference.forecast_inference_baseline(X_slow_train_concat, X_slow_train_trial_lengths, window_size)
                X_forecast_fast_train_baseline = utils_inference.forecast_inference_baseline(X_fast_train_concat, X_fast_train_trial_lengths, window_size)

                ## Baseine on test data
                X_forecast_slow_test_baseline = utils_inference.forecast_inference_baseline(X_slow_test_concat, X_slow_test_trial_lengths, window_size)
                X_forecast_fast_test_baseline = utils_inference.forecast_inference_baseline(X_fast_test_concat, X_fast_test_trial_lengths, window_size)

                ## Save inference and forecast results for later inspection
                forecast_slow_train[i_rs, i_fold, i_window] = X_forecast_slow_train_baseline
                forecast_slow_test[i_rs, i_fold, i_window]  = X_forecast_slow_test_baseline
                forecast_fast_train[i_rs, i_fold, i_window] = X_forecast_fast_train_baseline
                forecast_fast_test[i_rs, i_fold, i_window]  = X_forecast_fast_test_baseline

                ## Compute forecast error
                forecast_slow_train_baseline_rmse = utils_decoding.rmse(X_forecast_slow_train_baseline, X_slow_train_concat)
                forecast_fast_train_baseline_rmse = utils_decoding.rmse(X_forecast_fast_train_baseline, X_fast_train_concat)
                forecast_slow_test_baseline_rmse  = utils_decoding.rmse(X_forecast_slow_test_baseline,  X_slow_test_concat)
                forecast_fast_test_baseline_rmse  = utils_decoding.rmse(X_forecast_fast_test_baseline,  X_fast_test_concat)

                forecast_slow_train_baseline_r2 = utils_decoding.r2(X_forecast_slow_train_baseline, X_slow_train_concat)
                forecast_fast_train_baseline_r2 = utils_decoding.r2(X_forecast_fast_train_baseline, X_fast_train_concat)
                forecast_slow_test_baseline_r2  = utils_decoding.r2(X_forecast_slow_test_baseline,  X_slow_test_concat)
                forecast_fast_test_baseline_r2  = utils_decoding.r2(X_forecast_fast_test_baseline,  X_fast_test_concat)

                ## Save results (train index 0, test index 1)
                rmse_forecast_baseline_slow[0, i_rs, i_fold, i_window] = forecast_slow_train_baseline_rmse
                rmse_forecast_baseline_slow[1, i_rs, i_fold, i_window] = forecast_slow_test_baseline_rmse

                rmse_forecast_baseline_fast[0, i_rs, i_fold, i_window] = forecast_fast_train_baseline_rmse
                rmse_forecast_baseline_fast[1, i_rs, i_fold, i_window] = forecast_fast_test_baseline_rmse

                r2_forecast_baseline_slow[0, i_rs, i_fold, i_window] = forecast_slow_train_baseline_r2
                r2_forecast_baseline_slow[1, i_rs, i_fold, i_window] = forecast_slow_test_baseline_r2

                r2_forecast_baseline_fast[0, i_rs, i_fold, i_window] = forecast_fast_train_baseline_r2
                r2_forecast_baseline_fast[1, i_rs, i_fold, i_window] = forecast_fast_test_baseline_r2

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    
    ## Save results
    np.savez(
        res_save_path,
        rmse_forecast_baseline_slow=rmse_forecast_baseline_slow,
        rmse_forecast_baseline_fast=rmse_forecast_baseline_fast,
        r2_forecast_baseline_slow=r2_forecast_baseline_slow,
        r2_forecast_baseline_fast=r2_forecast_baseline_fast,
        # GT_slow=GT_slow,
        # GT_fast=GT_fast,
        # baseline_slow_test=baseline_slow_test,
        # baseline_fast_test=baseline_fast_test,
        random_states=random_states,
        n_folds=n_folds)



if __name__ == '__main__':

    for session_data_name in session_data_names:
        for unit_filter in unit_filters:
            for data_format in data_formats:

                print('=============================================================')
                print('Running Inference ...')
                print('\tsession_data_name: ', session_data_name)
                print('\tunit_filter: ',       unit_filter)
                print('\tdata_format: ',       data_format)
                print('\trandom_states: ',     random_states)
                print('\tn_folds: ',           n_folds)
                print('\twindow_sizes: ',      window_sizes)
                print('=============================================================')

                main(session_data_name, trial_filters, data_format)
