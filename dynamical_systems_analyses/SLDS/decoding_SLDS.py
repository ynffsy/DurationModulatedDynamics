"""Decode cursor trajectories from SLDS latent states across model configs."""

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



# -----------------------------------------------------------------------------
# Global configuration for exhaustive decoding sweeps
# -----------------------------------------------------------------------------
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
emission_classes   = config.emission_classes
init_types         = config.init_types
subspace_types     = config.subspace_types
alphas             = config.alphas



def main(
    session_data_name,
    unit_filter,
    input_unit_filter,
    data_format,
    label_format,
    train_test_option,
    model_type,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha):

    session_results_dir = os.path.join(results_dir, session_data_name)

    ## Initialize save name so each hyper-parameter tuple maps to one results file
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
        ns_discrete_states,
        ns_iters,
        model_type,
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))

    res_save_path = os.path.join(session_results_dir, res_save_name + '.npz')
    if os.path.exists(res_save_path) and not overwrite_results:
        print('Results already exist for file: ', res_save_path)
        return
    
    ## Load neural data for both movement speeds and requested preprocessing
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

    fast_model_results_dir, slow_model_results_dir = data_loader.get_model_result_dirs(
        time_offset=time_offset,
        data_format=data_format,
        train_test=train_test_option,
        model_type=model_type,
        dynamics_class=dynamics_class,
        emission_class=emission_class,
        init_type=init_type,
        subspace_type=subspace_type,
        alpha=alpha,
        check_existence=True)

    # data_loader.load_cursor_data()
    data_loader.align_cursor_to_firing_rates()
    cursor_states_fast, cursor_states_slow, _, _ = data_loader.reformat_cursor_data(label_format)

    if label_format == 'cartesian':
        metric = 'rmse'
    elif label_format in ['univec', 'pe']:
        metric = 'abs_angle'
        # metric = 'r2'
    elif label_format == 'pe_mag':
        metric = 'rel_mag'

    ## Initialize results
    ## (Train/test axis=0, random_states axis=1, folds axis=2, hyper-params afterwards)

    ## Decoding errors averaged over time points and trials
    decoding_errors_fast = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_discrete_states), len(ns_iters)))
    decoding_errors_slow = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_discrete_states), len(ns_iters)))

    ## Decoding errors per time point and trial (Order is preserved)
    trial_length_fast_max = np.max(trial_lengths_fast)
    trial_length_slow_max = np.max(trial_lengths_slow)

    decoding_errors_fast_test_per_time = np.full((len(random_states), len(ns_states), len(ns_discrete_states), len(ns_iters), n_trials_fast, trial_length_fast_max), np.nan, dtype=float)
    decoding_errors_slow_test_per_time = np.full((len(random_states), len(ns_states), len(ns_discrete_states), len(ns_iters), n_trials_slow, trial_length_slow_max), np.nan, dtype=float)

    ## Decoded results per trial
    decoded_results_fast_test = np.full((len(random_states), len(ns_states), len(ns_discrete_states), len(ns_iters), n_trials_fast, trial_length_fast_max, 2), np.nan, dtype=float)
    decoded_results_slow_test = np.full((len(random_states), len(ns_states), len(ns_discrete_states), len(ns_iters), n_trials_slow, trial_length_slow_max, 2), np.nan, dtype=float)
    
    decoding_errors_fast_test_per_trial = np.zeros((len(random_states), len(ns_states), len(ns_discrete_states), len(ns_iters), n_trials_fast))
    decoding_errors_slow_test_per_trial = np.zeros((len(random_states), len(ns_states), len(ns_discrete_states), len(ns_iters), n_trials_slow))


    ## Read SLDS results and decode cursor states
    for i_rs, random_state in enumerate(random_states):

        ## Multi-fold cross validation
        np.random.seed(random_state)

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('random state: ', random_state)

        ## K-fold cross validation (paired splits for slow/fast data)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        splits_slow = list(kf.split(np.arange(n_trials_slow)))
        splits_fast = list(kf.split(np.arange(n_trials_fast)))

        # Iterate through both splits using zip
        for i_fold, (
            (trial_indices_slow_train, trial_indices_slow_test), 
            (trial_indices_fast_train, trial_indices_fast_test)) in enumerate(zip(splits_slow, splits_fast)):
            
            print('fold: ', i_fold)

            if data_format is None:
                y_fast_train = [cursor_states_fast[i] for i in trial_indices_fast_train]
                y_fast_test  = [cursor_states_fast[i] for i in trial_indices_fast_test]
                y_slow_train = [cursor_states_slow[i] for i in trial_indices_slow_train]
                y_slow_test  = [cursor_states_slow[i] for i in trial_indices_slow_test]

            else:
                cursor_states_fast = np.array(cursor_states_fast)
                cursor_states_slow = np.array(cursor_states_slow)

                y_fast_train = cursor_states_fast[trial_indices_fast_train, :, :]
                y_fast_test  = cursor_states_fast[trial_indices_fast_test,  :, :]
                y_slow_train = cursor_states_slow[trial_indices_slow_train, :, :]
                y_slow_test  = cursor_states_slow[trial_indices_slow_test,  :, :]
                
            y_fast_train_concat = np.concatenate(y_fast_train, axis=0)
            y_fast_test_concat  = np.concatenate(y_fast_test,  axis=0)
            y_slow_train_concat = np.concatenate(y_slow_train, axis=0)
            y_slow_test_concat  = np.concatenate(y_slow_test,  axis=0)


            ## For SLDS, sweep through various numbers of states and EM iterations
            for i_continuous_states, n_continuous_states in enumerate(ns_states):
                for i_discrete_states, n_discrete_states in enumerate(ns_discrete_states):
                    for i_iters, n_iters in enumerate(ns_iters):

                        print('n_continuous_states: ', n_continuous_states, ' n_discrete_states: ', n_discrete_states, ' n_iters: ', n_iters)

                        ## Read SLDS processed data (latent states already inferred)
                        if model_type in ['LDS']:

                            ## Omit discrete states for LDS
                            model_save_name = '_'.join(map(str, [x for x in [
                                'r' + str(random_state),
                                'f' + str(i_fold),
                                's' + str(n_continuous_states),
                                'i' + str(n_iters)]]))
                        else:
                            model_save_name = '_'.join(map(str, [x for x in [
                                'r' + str(random_state),
                                'f' + str(i_fold),
                                's' + str(n_continuous_states),
                                'd' + str(n_discrete_states),
                                'i' + str(n_iters)]]))

                        slow_res_save_path = os.path.join(slow_model_results_dir, model_save_name + '.pkl')
                        fast_res_save_path = os.path.join(fast_model_results_dir, model_save_name + '.pkl')

                        with open(slow_res_save_path, 'rb') as f:
                            res_SLDS_slow = pickle.load(f)

                        with open(fast_res_save_path, 'rb') as f:
                            res_SLDS_fast = pickle.load(f)

                        continuous_states_slow_train = res_SLDS_slow['train_continuous_states']
                        continuous_states_slow_test  = res_SLDS_slow['test_continuous_states']
                        continuous_states_fast_train = res_SLDS_fast['train_continuous_states']
                        continuous_states_fast_test  = res_SLDS_fast['test_continuous_states']

                        X_slow_latent_slow_train = continuous_states_slow_train ## Slow latent space slow train data
                        X_fast_latent_fast_train = continuous_states_fast_train ## Fast latent space fast train data

                        assert len(y_slow_train) == len(X_slow_latent_slow_train)
                        assert len(y_fast_train) == len(X_fast_latent_fast_train)
                        X_slow_latent_slow_train_concat = np.concatenate(X_slow_latent_slow_train, axis=0)
                        X_fast_latent_fast_train_concat = np.concatenate(X_fast_latent_fast_train, axis=0)

                        if train_test_option in ['same_speed', 'joint']:
                            X_slow_latent_slow_test  = continuous_states_slow_test  ## Slow latent space slow test data
                            X_fast_latent_fast_test  = continuous_states_fast_test  ## Fast latent space fast test data

                            assert len(y_slow_test)  == len(X_slow_latent_slow_test)
                            assert len(y_fast_test)  == len(X_fast_latent_fast_test)
                            X_slow_latent_slow_test_concat = np.concatenate(X_slow_latent_slow_test, axis=0)
                            X_fast_latent_fast_test_concat = np.concatenate(X_fast_latent_fast_test, axis=0)

                        else:
                            X_slow_latent_fast_test  = continuous_states_slow_test  ## Slow latent space fast test data
                            X_fast_latent_slow_test  = continuous_states_fast_test  ## Fast latent space slow test data

                            assert len(y_slow_test)  == len(X_fast_latent_slow_test)
                            assert len(y_fast_test)  == len(X_slow_latent_fast_test)
                            X_slow_latent_fast_test_concat = np.concatenate(X_slow_latent_fast_test, axis=0)
                            X_fast_latent_slow_test_concat = np.concatenate(X_fast_latent_slow_test, axis=0)

                        ## Same-speed decoding
                        if train_test_option in ['same_speed', 'joint']:
                            
                            ## SLDS train slow test slow
                            rmse_train, rmse_test, e_train, e_test, train_pred, test_pred = utils_decoding.LR(
                                X_slow_latent_slow_train_concat, 
                                X_slow_latent_slow_test_concat, 
                                y_slow_train_concat, 
                                y_slow_test_concat,
                                metric)

                            decoding_errors_slow[0, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = rmse_train
                            decoding_errors_slow[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = rmse_test

                            # Store results per trial
                            cumulative_trial_length_slow = 0
                            for trial_id in trial_indices_slow_test:
                                trial_length = trial_lengths_slow[trial_id]
                                decoding_errors_slow_test_per_time[
                                    i_rs,
                                    i_continuous_states,
                                    i_discrete_states,
                                    i_iters,
                                    trial_id,
                                    :trial_length] = e_test[cumulative_trial_length_slow:cumulative_trial_length_slow + trial_length]
                                
                                decoded_results_slow_test[
                                    i_rs, 
                                    i_continuous_states, 
                                    i_discrete_states, 
                                    i_iters,
                                    trial_id,
                                    :trial_length] = test_pred[cumulative_trial_length_slow:cumulative_trial_length_slow + trial_length]
                                
                                decoding_errors_slow_test_per_trial[
                                    i_rs, 
                                    i_continuous_states, 
                                    i_discrete_states, 
                                    i_iters,
                                    trial_id] = np.mean(e_test[cumulative_trial_length_slow:cumulative_trial_length_slow + trial_length])
                                
                                cumulative_trial_length_slow += trial_length
                            

                            ## SLDS train fast test fast
                            rmse_train, rmse_test, e_train, e_test, train_pred, test_pred = utils_decoding.LR(
                                X_fast_latent_fast_train_concat, 
                                X_fast_latent_fast_test_concat, 
                                y_fast_train_concat, 
                                y_fast_test_concat,
                                metric)

                            decoding_errors_fast[0, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = rmse_train
                            decoding_errors_fast[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = rmse_test

                            # Store results per trial
                            cumulative_trial_length_fast = 0
                            for trial_id in trial_indices_fast_test:
                                trial_length = trial_lengths_fast[trial_id]
                                decoding_errors_fast_test_per_time[
                                    i_rs,
                                    i_continuous_states,
                                    i_discrete_states,
                                    i_iters,
                                    trial_id,
                                    :trial_length] = e_test[cumulative_trial_length_fast:cumulative_trial_length_fast + trial_length]
                                
                                decoded_results_fast_test[
                                    i_rs, 
                                    i_continuous_states, 
                                    i_discrete_states, 
                                    i_iters,
                                    trial_id,
                                    :trial_length] = test_pred[cumulative_trial_length_fast:cumulative_trial_length_fast + trial_length]
                                
                                decoding_errors_fast_test_per_trial[
                                    i_rs, 
                                    i_continuous_states, 
                                    i_discrete_states, 
                                    i_iters,
                                    trial_id] = np.mean(e_test[cumulative_trial_length_fast:cumulative_trial_length_fast + trial_length])
                                
                                cumulative_trial_length_fast += trial_length
                            
                        ## Cross-speed decoding
                        else:

                            ## SLDS train slow test fast                      
                            rmse_train, rmse_test, e_train, e_test, train_pred, test_pred = utils_decoding.LR(
                                X_slow_latent_slow_train_concat, 
                                X_slow_latent_fast_test_concat, 
                                y_slow_train_concat, 
                                y_fast_test_concat,
                                metric)
                            
                            decoding_errors_slow[0, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = rmse_train
                            decoding_errors_slow[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = rmse_test

                            # Store results per trial
                            cumulative_trial_length_fast = 0
                            for trial_id in trial_indices_fast_test:
                                trial_length = trial_lengths_fast[trial_id]
                                decoding_errors_fast_test_per_time[
                                    i_rs,
                                    i_continuous_states,
                                    i_discrete_states,
                                    i_iters,
                                    trial_id,
                                    :trial_length] = e_test[cumulative_trial_length_fast:cumulative_trial_length_fast + trial_length]
                                
                                decoded_results_fast_test[
                                    i_rs, 
                                    i_continuous_states, 
                                    i_discrete_states, 
                                    i_iters,
                                    trial_id,
                                    :trial_length] = test_pred[cumulative_trial_length_fast:cumulative_trial_length_fast + trial_length]
                                
                                decoding_errors_fast_test_per_trial[
                                    i_rs, 
                                    i_continuous_states, 
                                    i_discrete_states, 
                                    i_iters,
                                    trial_id] = np.mean(e_test[cumulative_trial_length_fast:cumulative_trial_length_fast + trial_length])
                                
                                cumulative_trial_length_fast += trial_length
                            

                            ## SLDS train fast test slow
                            rmse_train, rmse_test, e_train, e_test, train_pred, test_pred = utils_decoding.LR(
                                X_fast_latent_fast_train_concat, 
                                X_fast_latent_slow_test_concat, 
                                y_fast_train_concat, 
                                y_slow_test_concat,
                                metric)
                            
                            decoding_errors_fast[0, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = rmse_train
                            decoding_errors_fast[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = rmse_test
                            
                            # Store results per trial
                            cumulative_trial_length_slow = 0
                            for trial_id in trial_indices_slow_test:
                                trial_length = trial_lengths_slow[trial_id]
                                decoding_errors_slow_test_per_time[
                                    i_rs,
                                    i_continuous_states,
                                    i_discrete_states,
                                    i_iters,
                                    trial_id,
                                    :trial_length] = e_test[cumulative_trial_length_slow:cumulative_trial_length_slow + trial_length]
                                
                                decoded_results_slow_test[
                                    i_rs, 
                                    i_continuous_states, 
                                    i_discrete_states, 
                                    i_iters,
                                    trial_id,
                                    :trial_length] = test_pred[cumulative_trial_length_slow:cumulative_trial_length_slow + trial_length]
                                
                                decoding_errors_slow_test_per_trial[
                                    i_rs, 
                                    i_continuous_states, 
                                    i_discrete_states, 
                                    i_iters,
                                    trial_id] = np.mean(e_test[cumulative_trial_length_slow:cumulative_trial_length_slow + trial_length])
                                
                                cumulative_trial_length_slow += trial_length

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


    ## Save results (metrics, decoded trajectories, and trial length metadata)
    np.savez(
        res_save_path,
        decoding_errors_slow=decoding_errors_slow,
        decoding_errors_fast=decoding_errors_fast,
        decoding_errors_slow_test_per_time=decoding_errors_slow_test_per_time,
        decoding_errors_fast_test_per_time=decoding_errors_fast_test_per_time,
        decoding_errors_slow_test_per_trial=decoding_errors_slow_test_per_trial,
        decoding_errors_fast_test_per_trial=decoding_errors_fast_test_per_trial,
        decoded_results_slow_test=decoded_results_slow_test,
        decoded_results_fast_test=decoded_results_fast_test,
        trial_lengths_slow=trial_lengths_slow,
        trial_lengths_fast=trial_lengths_fast,)



if __name__ == '__main__':

    for (
        session_data_name, 
        unit_filter, 
        input_unit_filter, 
        data_format,
        label_format,
        train_test_option, 
        model_type, 
        dynamics_class, 
        emission_class,
        init_type, 
        subspace_type,
        alpha) in itertools.product(
            session_data_names, 
            unit_filters, 
            input_unit_filters, 
            data_formats,
            label_formats,
            train_test_options, 
            model_types, 
            dynamics_classes, 
            emission_classes,
            init_types, 
            subspace_types,
            alphas):

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
        print('\tns_discrete_states: ',   ns_discrete_states)
        print('\tns_iters: ',             ns_iters)
        print('\tmodel_type: ',           model_type) 
        print('\tdynamics_class: ',       dynamics_class)
        print('\temission_class: ',       emission_class)
        print('\tinit_type: ',            init_type)
        print('\tsubspace_type: ',        subspace_type)
        print('\talpha: ',                alpha)
        print('=============================================================')

        main(
            session_data_name, 
            unit_filter, 
            input_unit_filter, 
            data_format,
            label_format,
            train_test_option,
            model_type,
            dynamics_class,
            emission_class,
            init_type,
            subspace_type,
            alpha)
