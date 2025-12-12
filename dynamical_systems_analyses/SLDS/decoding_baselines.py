import os
import time
import ipdb
import pickle
import itertools
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import dynamical_systems_analyses.SLDS.config as config
import dynamical_systems_analyses.utils.utils_processing as utils_processing
import dynamical_systems_analyses.utils.utils_decoding as utils_decoding
from vis_config import session_target_radii



## Read parameters from config
overwrite_results  = config.overwrite_results
data_dir           = config.data_dir
results_dir        = config.results_dir
session_data_names = config.session_data_names

unit_filters       = config.unit_filters
input_unit_filters = config.input_unit_filters
window_config      = config.window_config
time_offset        = config.time_offset
data_formats       = config.data_formats
label_formats      = config.label_formats
trial_filters      = config.trial_filters
train_test_options = config.train_test_options

random_states      = config.random_states
n_folds            = config.n_folds
ns_states          = config.ns_states
ns_discrete_states = config.ns_discrete_states
ns_iters           = config.ns_iters

model_types        = config.model_types
dynamics_classes   = config.dynamics_classes
init_types         = config.init_types
alphas             = config.alphas



def main(
    session_data_name,
    unit_filter,
    input_unit_filter,
    data_format,
    label_format,
    train_test_option):

    session_results_dir = os.path.join(results_dir, session_data_name)

    ## Initialize save name
    res_save_name = '_'.join(map(str, [x for x in [
        'decoding',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        label_format,
        trial_filters,
        train_test_option,
        random_states,
        n_folds,
        ns_states,
        'PCA'] if x is not None]))

    res_save_path = os.path.join(session_results_dir, res_save_name + '.npz')
    if os.path.exists(res_save_path) and not overwrite_results:
        print('Results already exist for file: ', res_save_path)
        return
    
    ## Load data
    data_loader = utils_processing.DataLoaderDuo(
        data_dir,
        results_dir,
        session_data_name,
        unit_filter,
        input_unit_filter,
        window_config,
        trial_filters)
    
    data_loader.load_firing_rate_data()
    data_loader.load_cursor_data()
    data_loader.remove_target_overlap(target_radius=session_target_radii[session_data_name])

    (firing_rates_fast_simple, firing_rates_slow_simple, 
     _, _,
     _, _, 
     trial_lengths_fast, trial_lengths_slow,
     _, _,
     _, _) = data_loader.reformat_firing_rate_data(data_format, trial_length_filter_percentile=90)
    
    n_trials_fast = len(firing_rates_fast_simple)
    n_trials_slow = len(firing_rates_slow_simple)

    print('n_trials_fast: ', n_trials_fast, 'n_trials_slow: ', n_trials_slow)

    # data_loader.load_cursor_data()
    data_loader.align_cursor_to_firing_rates()
    cursor_states_fast, cursor_states_slow, _, _ = data_loader.reformat_cursor_data(label_format)

    if label_format == 'cartesian':
        metric = 'rmse'
    elif label_format in ['univec', 'pe']:
        metric = 'abs_angle'
    elif label_format == 'pe_mag':
        metric = 'rel_mag'

    ## Initialize results
    ## Cannot save decoding results for training data because results from each fold would overwrite each other
    
    ## Decoding errors averaged over time points and trials
    decoding_errors_slow = np.zeros((2, len(random_states), n_folds, len(ns_states)))
    decoding_errors_fast = np.zeros((2, len(random_states), n_folds, len(ns_states)))

    ## Decoding errors per time point and trial
    decoding_errors_slow_test_per_time = np.zeros((len(random_states), len(ns_states), np.sum(trial_lengths_slow)))
    decoding_errors_fast_test_per_time = np.zeros((len(random_states), len(ns_states), np.sum(trial_lengths_fast)))

    ## Decoded results per time point
    decoded_results_slow_test = np.empty((len(random_states), n_trials_slow, len(ns_states)), dtype=object)
    decoded_results_fast_test = np.empty((len(random_states), n_trials_fast, len(ns_states)), dtype=object)

    ## Read SLDS results and decode cursor states
    for i_rs, random_state in enumerate(random_states):

        ## Multi-fold cross validation
        np.random.seed(random_state)

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('random state: ', random_state)

        ## K-fold cross validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        # for i_fold, (trial_indices_train, trial_indices_test) in enumerate(kf.split(np.arange(n_trials_slow))):

        splits_slow = list(kf.split(np.arange(n_trials_slow)))
        splits_fast = list(kf.split(np.arange(n_trials_fast)))

        cumulative_trial_lengths_slow = 0
        cumulative_trial_lengths_fast = 0

        # Iterate through both splits using zip
        for i_fold, (
            (trial_indices_slow_train, trial_indices_slow_test), 
            (trial_indices_fast_train, trial_indices_fast_test)) in enumerate(zip(splits_slow, splits_fast)):
            
            print('fold: ', i_fold)

            if data_format is None:
                X_slow_train = [firing_rates_slow_simple[i] for i in trial_indices_slow_train]
                X_slow_test  = [firing_rates_slow_simple[i] for i in trial_indices_slow_test]
                X_fast_train = [firing_rates_fast_simple[i] for i in trial_indices_fast_train]
                X_fast_test  = [firing_rates_fast_simple[i] for i in trial_indices_fast_test]
                    
                y_slow_train = [cursor_states_slow[i] for i in trial_indices_slow_train]
                y_slow_test  = [cursor_states_slow[i] for i in trial_indices_slow_test]
                y_fast_train = [cursor_states_fast[i] for i in trial_indices_fast_train]
                y_fast_test  = [cursor_states_fast[i] for i in trial_indices_fast_test]

            else:
                cursor_states_slow = np.array(cursor_states_slow)
                cursor_states_fast = np.array(cursor_states_fast)

                X_slow_train = firing_rates_slow_simple[trial_indices_slow_train, :, :]
                X_slow_test  = firing_rates_slow_simple[trial_indices_slow_test,  :, :]
                X_fast_train = firing_rates_fast_simple[trial_indices_fast_train, :, :]
                X_fast_test  = firing_rates_fast_simple[trial_indices_fast_test,  :, :]

                y_slow_train = cursor_states_slow[trial_indices_slow_train, :, :]
                y_slow_test  = cursor_states_slow[trial_indices_slow_test,  :, :]
                y_fast_train = cursor_states_fast[trial_indices_fast_train, :, :]
                y_fast_test  = cursor_states_fast[trial_indices_fast_test,  :, :]
                
            X_slow_train_concat = np.concatenate(X_slow_train, axis=0)
            X_slow_test_concat  = np.concatenate(X_slow_test,  axis=0)
            X_fast_train_concat = np.concatenate(X_fast_train, axis=0)
            X_fast_test_concat  = np.concatenate(X_fast_test,  axis=0)
            
            y_slow_train_concat = np.concatenate(y_slow_train, axis=0)
            y_slow_test_concat  = np.concatenate(y_slow_test,  axis=0)
            y_fast_train_concat = np.concatenate(y_fast_train, axis=0)
            y_fast_test_concat  = np.concatenate(y_fast_test,  axis=0)

            fold_total_trial_lengths_slow = np.sum(trial_lengths_slow[trial_indices_slow_test])
            fold_total_trial_lengths_fast = np.sum(trial_lengths_fast[trial_indices_fast_test])

            ## For SLDS, sweep through various numbers of states and iterations
            for i_continuous_states, n_continuous_states in enumerate(ns_states):

                print('n_continuous_states: ', n_continuous_states)

                ## Same-speed decoding
                if train_test_option == 'same_speed':
                    
                    ## SLDS train slow test slow
                    rmse_train, rmse_test, e_train, e_test, train_pred, test_pred = utils_decoding.LR(
                        X_slow_train_concat, 
                        X_slow_test_concat, 
                        y_slow_train_concat, 
                        y_slow_test_concat,
                        metric=metric,
                        n_components=n_continuous_states)

                    decoding_errors_slow[0, i_rs, i_fold, i_continuous_states] = rmse_train
                    decoding_errors_slow[1, i_rs, i_fold, i_continuous_states] = rmse_test

                    decoding_errors_slow_test_per_time[
                        i_rs, 
                        i_continuous_states, 
                        cumulative_trial_lengths_slow : cumulative_trial_lengths_slow + fold_total_trial_lengths_slow] = e_test

                    ## Convert test_pred to list of arrays
                    total_length = 0
                    for trial_index in trial_indices_slow_test:
                        trial_length = trial_lengths_slow[trial_index]
                        decoded_results_slow_test[i_rs, trial_index, i_continuous_states] = test_pred[total_length:total_length + trial_length]
                        total_length += trial_length

                    ## SLDS train fast test fast
                    rmse_train, rmse_test, e_train, e_test, train_pred, test_pred = utils_decoding.LR(
                        X_fast_train_concat, 
                        X_fast_test_concat, 
                        y_fast_train_concat, 
                        y_fast_test_concat,
                        metric=metric,
                        n_components=n_continuous_states)
                    
                    decoding_errors_fast[0, i_rs, i_fold, i_continuous_states] = rmse_train
                    decoding_errors_fast[1, i_rs, i_fold, i_continuous_states] = rmse_test

                    decoding_errors_fast_test_per_time[
                        i_rs, 
                        i_continuous_states, 
                        cumulative_trial_lengths_fast : cumulative_trial_lengths_fast + fold_total_trial_lengths_fast] = e_test
                    
                    ## Convert test_pred to list of arrays
                    total_length = 0
                    for trial_index in trial_indices_fast_test:
                        trial_length = trial_lengths_fast[trial_index]
                        decoded_results_fast_test[i_rs, trial_index, i_continuous_states] = test_pred[total_length:total_length + trial_length]
                        total_length += trial_length

                ## Cross-speed decoding
                else:

                    ## SLDS train slow test fast                      
                    rmse_train, rmse_test, e_train, e_test, train_pred, test_pred = utils_decoding.LR(
                        X_slow_train_concat, 
                        X_fast_test_concat, 
                        y_slow_train_concat, 
                        y_fast_test_concat,
                        metric=metric,
                        n_components=n_continuous_states)
                    
                    decoding_errors_slow[0, i_rs, i_fold, i_continuous_states] = rmse_train
                    decoding_errors_slow[1, i_rs, i_fold, i_continuous_states] = rmse_test

                    decoding_errors_fast_test_per_time[
                        i_rs, 
                        i_continuous_states, 
                        cumulative_trial_lengths_fast : cumulative_trial_lengths_fast + fold_total_trial_lengths_fast] = e_test
                    
                    ## Convert test_pred to list of arrays
                    total_length = 0
                    for trial_index in trial_indices_fast_test:
                        trial_length = trial_lengths_fast[trial_index]
                        decoded_results_fast_test[i_rs, trial_index, i_continuous_states] = test_pred[total_length:total_length + trial_length]
                        total_length += trial_length

                    ## SLDS train fast test slow
                    rmse_train, rmse_test, e_train, e_test, train_pred, test_pred = utils_decoding.LR(
                        X_fast_train_concat, 
                        X_slow_test_concat, 
                        y_fast_train_concat, 
                        y_slow_test_concat,
                        metric=metric,
                        n_components=n_continuous_states)
                    
                    decoding_errors_fast[0, i_rs, i_fold, i_continuous_states] = rmse_train
                    decoding_errors_fast[1, i_rs, i_fold, i_continuous_states] = rmse_test

                    decoding_errors_slow_test_per_time[
                        i_rs, 
                        i_continuous_states, 
                        cumulative_trial_lengths_slow : cumulative_trial_lengths_slow + fold_total_trial_lengths_slow] = e_test
                    
                    ## Convert test_pred to list of arrays
                    total_length = 0
                    for trial_index in trial_indices_slow_test:
                        trial_length = trial_lengths_slow[trial_index]
                        decoded_results_slow_test[i_rs, trial_index, i_continuous_states] = test_pred[total_length:total_length + trial_length]
                        total_length += trial_length

            
            cumulative_trial_lengths_slow += fold_total_trial_lengths_slow
            cumulative_trial_lengths_fast += fold_total_trial_lengths_fast

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


    ## Save results
    np.savez(
        res_save_path,
        decoding_errors_slow=decoding_errors_slow,
        decoding_errors_fast=decoding_errors_fast,
        decoding_errors_slow_test_per_time=decoding_errors_slow_test_per_time,
        decoding_errors_fast_test_per_time=decoding_errors_fast_test_per_time,
        decoded_results_slow_test=decoded_results_slow_test,
        decoded_results_fast_test=decoded_results_fast_test)



if __name__ == '__main__':

    for (
        session_data_name, 
        unit_filter, 
        input_unit_filter, 
        data_format,
        label_format,
        train_test_option) in itertools.product(
            session_data_names, 
            unit_filters, 
            input_unit_filters, 
            data_formats,
            label_formats,
            train_test_options):

        print('=============================================================')
        print('Running Decoding ...')
        print('\tsession_data_name: ',    session_data_name)
        print('\tunit_filter: ',          unit_filter)
        print('\tinput_unit_filter: ',    input_unit_filter)
        print('\tdata_format: ',          data_format)
        print('\tlabel_format: ',         label_format)
        print('\ttrial_filters: ',        trial_filters)
        print('\ttrain_test_option: ',    train_test_option)
        print('\trandom_states: ',        random_states)
        print('\tn_folds: ',              n_folds)
        print('\tns_continuous_states: ', ns_states)
        print('=============================================================')

        main(
            session_data_name, 
            unit_filter, 
            input_unit_filter, 
            data_format,
            label_format,
            train_test_option)
