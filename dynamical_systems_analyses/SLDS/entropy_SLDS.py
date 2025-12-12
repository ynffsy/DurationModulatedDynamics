"""Compute discrete-state entropy/transition penalties from SLDS fits."""

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
import dynamical_systems_analyses.utils.utils_vis as utils_vis
from vis_config import session_target_radii



# -----------------------------------------------------------------------------
# Global configuration shared across all entropy jobs
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
    train_test_option,
    model_type,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    discrete_state_reformat='resample'):

    session_results_dir = os.path.join(results_dir, session_data_name)

    ## Initialize save name for this unique combination of hyper-parameters
    res_save_name = '_'.join(map(str, [x for x in [
        'entropy',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
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
    
    ## Load firing rates/cursor data for the requested session
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

    ## Initialize results containers
    entropy_per_time_fast = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_discrete_states), len(ns_iters)))
    entropy_per_time_slow = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_discrete_states), len(ns_iters)))
    entropy_per_trial_fast = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_discrete_states), len(ns_iters)))
    entropy_per_trial_slow = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_discrete_states), len(ns_iters)))
    transition_penalty_per_trial_fast = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_discrete_states), len(ns_iters)))
    transition_penalty_per_trial_slow = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_discrete_states), len(ns_iters)))

    ## Read SLDS results and decode cursor states
    for i_rs, random_state in enumerate(random_states):

        ## Multi-fold cross validation
        np.random.seed(random_state)

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('random state: ', random_state)

        ## K-fold cross validation so each random state sees the same folds
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        # for i_fold, (trial_indices_train, trial_indices_test) in enumerate(kf.split(np.arange(n_trials_slow))):
        splits_fast = list(kf.split(np.arange(n_trials_fast)))
        splits_slow = list(kf.split(np.arange(n_trials_slow)))

        # Iterate through both splits using zip
        for i_fold, (
            (trial_indices_slow_train, trial_indices_slow_test), 
            (trial_indices_fast_train, trial_indices_fast_test)) in enumerate(zip(splits_slow, splits_fast)):
            
            print('fold: ', i_fold)

            ## For SLDS, sweep through various numbers of states and iterations
            for i_continuous_states, n_continuous_states in enumerate(ns_states):
                for i_discrete_states, n_discrete_states in enumerate(ns_discrete_states):
                    for i_iters, n_iters in enumerate(ns_iters):

                        if n_discrete_states == 1:
                            continue

                        print('n_continuous_states: ', n_continuous_states, ' n_discrete_states: ', n_discrete_states, ' n_iters: ', n_iters)

                        ## Read SLDS processed data
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

                        fast_res_save_path = os.path.join(fast_model_results_dir, model_save_name + '.pkl')
                        slow_res_save_path = os.path.join(slow_model_results_dir, model_save_name + '.pkl')
                        
                        with open(fast_res_save_path, 'rb') as f:
                            res_SLDS_fast = pickle.load(f)
                        with open(slow_res_save_path, 'rb') as f:
                            res_SLDS_slow = pickle.load(f)

                        ## Extract discrete states
                        discrete_states_fast_train = res_SLDS_fast['train_discrete_states']
                        discrete_states_fast_test  = res_SLDS_fast['test_discrete_states']
                        discrete_states_slow_train = res_SLDS_slow['train_discrete_states']
                        discrete_states_slow_test  = res_SLDS_slow['test_discrete_states']

                        ## If trials have different lengths, reformat the discrete states
                        if data_format is None:
                            if discrete_state_reformat == 'truncate_end':
                                discrete_states_fast_train_reformat = utils_processing.truncate_discrete_states(discrete_states_fast_train, truncate_end=True)
                                discrete_states_slow_train_reformat = utils_processing.truncate_discrete_states(discrete_states_slow_train, truncate_end=True)
                                discrete_states_fast_test_reformat  = utils_processing.truncate_discrete_states(discrete_states_fast_test,  truncate_end=True)
                                discrete_states_slow_test_reformat  = utils_processing.truncate_discrete_states(discrete_states_slow_test,  truncate_end=True)
                            elif discrete_state_reformat == 'truncate_front':
                                discrete_states_fast_train_reformat = utils_processing.truncate_discrete_states(discrete_states_fast_train, truncate_end=False)
                                discrete_states_slow_train_reformat = utils_processing.truncate_discrete_states(discrete_states_slow_train, truncate_end=False)
                                discrete_states_fast_test_reformat  = utils_processing.truncate_discrete_states(discrete_states_fast_test,  truncate_end=False)
                                discrete_states_slow_test_reformat  = utils_processing.truncate_discrete_states(discrete_states_slow_test,  truncate_end=False)
                            elif discrete_state_reformat == 'resample':
                                discrete_states_fast_train_reformat = utils_processing.resample_discrete_states(discrete_states_fast_train)
                                discrete_states_slow_train_reformat = utils_processing.resample_discrete_states(discrete_states_slow_train)
                                discrete_states_fast_test_reformat  = utils_processing.resample_discrete_states(discrete_states_fast_test)
                                discrete_states_slow_test_reformat  = utils_processing.resample_discrete_states(discrete_states_slow_test)
                            else:
                                raise ValueError('discrete_state_reformat not recognized')
                        else:
                            discrete_states_fast_train_reformat = discrete_states_fast_train
                            discrete_states_slow_train_reformat = discrete_states_slow_train
                            discrete_states_fast_test_reformat  = discrete_states_fast_test
                            discrete_states_slow_test_reformat  = discrete_states_slow_test
                        
                        ## Compute entropy
                        entropy_per_time_fast_train = utils_vis.entropy_per_time_point_lumped(discrete_states_fast_train_reformat, num_classes=n_discrete_states)
                        entropy_per_time_fast_test  = utils_vis.entropy_per_time_point_lumped(discrete_states_fast_test_reformat,  num_classes=n_discrete_states)
                        entropy_per_time_slow_train = utils_vis.entropy_per_time_point_lumped(discrete_states_slow_train_reformat, num_classes=n_discrete_states)
                        entropy_per_time_slow_test  = utils_vis.entropy_per_time_point_lumped(discrete_states_slow_test_reformat,  num_classes=n_discrete_states)
                        
                        entropy_per_trial_fast_train = utils_vis.entropy_per_trial(discrete_states_fast_train, num_classes=n_discrete_states)
                        entropy_per_trial_fast_test  = utils_vis.entropy_per_trial(discrete_states_fast_test,  num_classes=n_discrete_states)
                        entropy_per_trial_slow_train = utils_vis.entropy_per_trial(discrete_states_slow_train, num_classes=n_discrete_states)
                        entropy_per_trial_slow_test  = utils_vis.entropy_per_trial(discrete_states_slow_test,  num_classes=n_discrete_states)
                        
                        transition_penalty_per_trial_fast_train = utils_vis.transition_penalty_per_trial(discrete_states_fast_train)
                        transition_penalty_per_trial_fast_test  = utils_vis.transition_penalty_per_trial(discrete_states_fast_test)
                        transition_penalty_per_trial_slow_train = utils_vis.transition_penalty_per_trial(discrete_states_slow_train)
                        transition_penalty_per_trial_slow_test  = utils_vis.transition_penalty_per_trial(discrete_states_slow_test)

                        ## Average over time
                        entropy_per_time_fast_train = np.mean(entropy_per_time_fast_train)
                        entropy_per_time_fast_test  = np.mean(entropy_per_time_fast_test)
                        entropy_per_time_slow_train = np.mean(entropy_per_time_slow_train)
                        entropy_per_time_slow_test  = np.mean(entropy_per_time_slow_test)

                        entropy_per_trial_fast_train = np.mean(entropy_per_trial_fast_train)
                        entropy_per_trial_fast_test  = np.mean(entropy_per_trial_fast_test)
                        entropy_per_trial_slow_train = np.mean(entropy_per_trial_slow_train)
                        entropy_per_trial_slow_test  = np.mean(entropy_per_trial_slow_test)
                        

                        transition_penalty_per_trial_fast_train = np.mean(transition_penalty_per_trial_fast_train)
                        transition_penalty_per_trial_fast_test  = np.mean(transition_penalty_per_trial_fast_test)
                        transition_penalty_per_trial_slow_train = np.mean(transition_penalty_per_trial_slow_train)
                        transition_penalty_per_trial_slow_test  = np.mean(transition_penalty_per_trial_slow_test)

                        ## entropy slow means the model was trained on slow trials, both same speed and cross speed
                        entropy_per_time_fast[0, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = entropy_per_time_fast_train
                        entropy_per_time_slow[0, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = entropy_per_time_slow_train
                        
                        entropy_per_trial_fast[0, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = entropy_per_trial_fast_train
                        entropy_per_trial_slow[0, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = entropy_per_trial_slow_train

                        transition_penalty_per_trial_fast[0, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = transition_penalty_per_trial_fast_train
                        transition_penalty_per_trial_slow[0, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = transition_penalty_per_trial_slow_train
                        
                        
                        ## Same-speed
                        if train_test_option == 'same_speed':
                            entropy_per_time_fast[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = entropy_per_time_fast_test
                            entropy_per_time_slow[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = entropy_per_time_slow_test
                            
                            entropy_per_trial_fast[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = entropy_per_trial_fast_test
                            entropy_per_trial_slow[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = entropy_per_trial_slow_test
                            
                            transition_penalty_per_trial_fast[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = transition_penalty_per_trial_fast_test
                            transition_penalty_per_trial_slow[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = transition_penalty_per_trial_slow_test
                        
                        ## Cross-speed
                        else:
                            entropy_per_time_fast[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = entropy_per_time_slow_test
                            entropy_per_time_slow[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = entropy_per_time_fast_test
                            
                            entropy_per_trial_fast[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = entropy_per_trial_slow_test
                            entropy_per_trial_slow[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = entropy_per_trial_fast_test
                            
                            transition_penalty_per_trial_fast[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = transition_penalty_per_trial_slow_test
                            transition_penalty_per_trial_slow[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = transition_penalty_per_trial_fast_test

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


    # Save results
    np.savez(
        res_save_path,
        entropy_per_time_fast=entropy_per_time_fast,
        entropy_per_time_slow=entropy_per_time_slow,
        entropy_per_trial_fast=entropy_per_trial_fast,
        entropy_per_trial_slow=entropy_per_trial_slow,
        transition_penalty_per_trial_fast=transition_penalty_per_trial_fast,
        transition_penalty_per_trial_slow=transition_penalty_per_trial_slow)
        


if __name__ == '__main__':

    for (
        session_data_name, 
        unit_filter, 
        input_unit_filter, 
        data_format,
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
            train_test_options, 
            model_types, 
            dynamics_classes, 
            emission_classes,
            init_types, 
            subspace_types,
            alphas):

        print('=============================================================')
        print('Computing Entropy for discrete states...')
        print('\tsession_data_name: ',    session_data_name)
        print('\tunit_filter: ',          unit_filter)
        print('\tinput_unit_filter: ',    input_unit_filter)
        print('\tdata_format: ',          data_format)
        print('\ttrial_filters: ',        trial_filters)
        print('\train_test_option: ',     train_test_option)
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
            train_test_option,
            model_type,
            dynamics_class,
            emission_class,
            init_type,
            subspace_type,
            alpha)
