"""Cross-validated SLDS training for a single speed condition.

Splits trials within one behavioral filter into K folds, fits SLDS models
for each hyper-parameter combination, and saves latents for train/test
segments across folds.
"""

import os
import time
import ipdb
import pickle
import datetime
import itertools

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import dynamical_systems_analyses.SLDS.config as config
import dynamical_systems_analyses.utils.utils_processing as utils_processing
from SLDS import SLDS
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
trial_filters      = config.trial_filters

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
    trial_filter,
    model_type,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha):

    """Run K-fold SLDS training for one session/condition pairing."""
    
    ## Load data
    data_loader = utils_processing.DataLoader(
        data_dir,
        results_dir,
        session_data_name,
        unit_filter,
        input_unit_filter,
        window_config,
        trial_filter)
    
    data_loader.load_firing_rate_data()
    data_loader.load_cursor_data()
    data_loader.remove_target_overlap(target_radius=session_target_radii[session_data_name])
    
    # if 'masked' in trial_filter:
    #     data_loader.compute_masked_trial_cutoff_times(
    #         time_limit=vis.masked_trials_time_limits[trial_filter], 
    #         distance_limit=vis.masked_trials_distance_limit)
            
    (firing_rates_simple, 
     input_firing_rates_simple,
     _, 
     _, 
     _, 
     _) = data_loader.reformat_firing_rate_data(data_format, time_offset=time_offset, zero_padding=True, trial_length_filter_percentile=90)
    
    n_trials = len(firing_rates_simple)
    print('n_trials: ', n_trials)

    n_neurons = firing_rates_simple[0].shape[1]

    if input_firing_rates_simple is not None:
        n_input_neurons = input_firing_rates_simple[0].shape[1]
    else:
        n_input_neurons = 0
    
    model_results_dir = data_loader.get_model_result_dir(
        time_offset=time_offset,
        data_format=data_format,
        train_test='same_speed',
        model_type=model_type,
        dynamics_class=dynamics_class,
        emission_class=emission_class,
        init_type=init_type,
        subspace_type=subspace_type,
        alpha=alpha,
        check_existence=False)
    
    print('model_results_dir: ', model_results_dir)
    
    ## Training with cross-validation
    for i_rs, random_state in enumerate(random_states):

        ## Multi-fold cross validation
        np.random.seed(random_state)

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('random state: ', random_state)

        ## Unbalanced sampling
        # fold_indices = np.random.randint(n_folds, size=n_trials)
        # print(fold_indices)

        ## K-fold cross validation over all trials within the filter
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        # for i_fold in range(n_folds):
        for i_fold, (trial_indices_train, trial_indices_test) in enumerate(kf.split(np.arange(n_trials))):
            
            print('fold: ', i_fold)
            print('trial_indices_test: ', trial_indices_test)

            if data_format in ['fill0', 'truncate_front', 'truncate_end', 'resample']:
                X_train = firing_rates_simple[trial_indices_train, :, :]
                X_test  = firing_rates_simple[trial_indices_test, :, :]

                if input_firing_rates_simple is not None:
                    X_input_train = np.array(input_firing_rates_simple)[trial_indices_train, :, :]
                    X_input_test  = np.array(input_firing_rates_simple)[trial_indices_test, :, :]
                else:
                    X_input_train = None
                    X_input_test  = None

            else:
                X_train = [firing_rates_simple[i] for i in trial_indices_train]
                X_test  = [firing_rates_simple[i] for i in trial_indices_test]

                if input_firing_rates_simple is not None:
                    X_input_train = [input_firing_rates_simple[i] for i in trial_indices_train]
                    X_input_test  = [input_firing_rates_simple[i] for i in trial_indices_test]
                else:
                    X_input_train = None
                    X_input_test  = None

            ## Sweep through all requested state and iteration counts
            for i_continuous_states, n_continuous_states in enumerate(ns_states):
                for i_discrete_states, n_discrete_states in enumerate(ns_discrete_states):
                    for i_iters, n_iters in enumerate(ns_iters):

                        time_start = time.time()

                        ## Format save path encoding seed/fold/state counts
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

                        model_save_prefix = os.path.join(model_results_dir, model_save_name)
                        res_save_path     = model_save_prefix + '.pkl'

                        if (not overwrite_results) and os.path.isfile(res_save_path):
                            print('s' + str(n_continuous_states) + 'i' + str(n_iters) + ' model already exists. Skipping...')
                            continue

                        ## Use SLDS to reduce the dimensionality of firing rate data                    
                        neural_SLDS = SLDS(
                            X_train,
                            X_input_train,
                            n_neurons,
                            n_input_neurons,
                            data_format,
                            random_state,
                            n_continuous_states,
                            n_discrete_states,
                            n_iters,
                            model_type,
                            dynamics_class,
                            emission_class,
                            init_type,
                            subspace_type,
                            alpha)

                        neural_SLDS.fit()
                        neural_SLDS.transform(
                            test_emissions=X_test, 
                            test_inputs=X_input_test)

                        with open(res_save_path , 'wb') as f:
                            pickle.dump({
                                'train_elbos'                        : neural_SLDS.train_elbos,
                                'train_continuous_states'            : neural_SLDS.train_continuous_states,
                                'train_continuous_state_covariances' : neural_SLDS.train_continuous_state_covariances,
                                'train_discrete_states'              : neural_SLDS.train_discrete_states,
                                'test_elbos'                         : neural_SLDS.test_elbos,
                                'test_continuous_states'             : neural_SLDS.test_continuous_states,
                                'test_continuous_state_covariances'  : neural_SLDS.test_continuous_state_covariances,
                                'test_discrete_states'               : neural_SLDS.test_discrete_states,
                                'model'                              : neural_SLDS.model,
                            }, f)

                        time_end = time.time()
                        runtime = time_end - time_start
                        # runtimes[i_rs, i_fold, i_states, i_iters] = runtime

                        print('# continuous states: ', n_continuous_states, ' # discrete states: ', n_discrete_states, ' # iters: ', n_iters, ' run time: ', runtime)

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    
    ## Save runtimes and trial indices for each fold
    ## (Not saving the results when all models are skipped)
    # if not np.all(runtimes == 0):
        
    #     metadata_save_name = 'metadata_' + datetime.datetime.now().strftime('%m%d%Y-%H%M%S')    
    #     metadata_save_path = os.path.join(model_results_dir, metadata_save_name + '.pkl')

    #     with open(metadata_save_path, 'wb') as f:
    #         pickle.dump({
    #             'n_neurons'     : n_neurons,
    #             'n_trials'      : n_trials,
    #             'random_states' : random_states,
    #             'n_folds'       : n_folds,
    #             'ns_states'     : ns_states,
    #             'ns_iters'      : ns_iters,
    #             'alpha'         : alpha,
    #             'data_format'   : data_format,
    #             'runtimes'      : runtimes,
    #             'fold_indices'  : fold_indices,
    #         }, f)



if __name__ == '__main__':

    for (
        session_data_name, 
        unit_filter, 
        input_unit_filter, 
        data_format,
        trial_filter, 
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
            trial_filters, 
            model_types, 
            dynamics_classes, 
            emission_classes,
            init_types, 
            subspace_types,
            alphas):

        print('=============================================================')
        print('Running Same Speed SLDS')
        print('\tsession_data_name: ',    session_data_name)
        print('\tunit_filter: ',          unit_filter)
        print('\tinput_unit_filter: ',    input_unit_filter)
        print('\twindow_config: ',        window_config)
        print('\tdata_format: ',          data_format)
        print('\ttrial_filter: ',         trial_filter)
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
            trial_filter, 
            model_type, 
            dynamics_class,
            emission_class,
            init_type, 
            subspace_type,
            alpha)
