"""Full cross-speed SLDS training loop with K-fold cross-validation.

Pairs each behavioral speed with its counterpart, fits SLDS models on the
training folds, and evaluates on held-out trials to capture generalization
across both trial splits and speed conditions.
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

import scripts.config as config
import utils.utils_processing as utils_processing
from experiments.SLDS import SLDS

        

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

    """Fit/evaluate SLDS models for a single configuration tuple."""

    trial_filter_ctpt = utils_processing.trial_filter_counterparts[trial_filter]

    ## Load data
    data_loader = utils_processing.DataLoaderDuo(
        data_dir,
        results_dir,
        session_data_name,
        unit_filter,
        input_unit_filter,
        window_config,
        [trial_filter, trial_filter_ctpt])
    
    data_loader.load_firing_rate_data()

    ## Get paired data for self and ctpt (counterpart) trials
    (firing_rates_self_simple,       firing_rates_ctpt_simple, 
     input_firing_rates_self_simple, input_firing_rates_ctpt_simple,
     trial_ids_self,                 trial_ids_ctpt, 
     n_trials_self,                  n_trials_ctpt,
     trial_lengths_self,             trial_lengths_ctpt,
     times_new_self,                 times_new_ctpt) = data_loader.reformat_firing_rate_data(data_format)
    
    n_neurons = firing_rates_self_simple[0].shape[1]
    assert n_neurons == firing_rates_ctpt_simple[0].shape[1]

    if input_firing_rates_self_simple is not None:
        n_input_neurons = input_firing_rates_self_simple[0].shape[1]
        assert n_input_neurons == input_firing_rates_ctpt_simple[0].shape[1]
    else:
        n_input_neurons = 0

    model_results_dir, _ = data_loader.get_model_result_dirs(
        time_offset=time_offset,
        data_format=data_format,
        train_test='cross_speed',
        model_type=model_type,
        dynamics_class=dynamics_class,
        emission_class=emission_class,
        init_type=init_type,
        subspace_type=subspace_type,
        alpha=alpha,
        check_existence=False)
    
    print ('model_results_dir: ', model_results_dir)

    ## Save runtimes and trial indices for each fold
    runtimes = np.zeros((len(random_states), n_folds, len(ns_states), len(ns_iters)))

    ## Training with cross-validation
    for i_rs, random_state in enumerate(random_states):

        ## Multi-fold cross validation
        np.random.seed(random_state)

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('random state: ', random_state)

        ## K-fold cross validation across self vs counterpart trials
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        splits_self = list(kf.split(np.arange(n_trials_self)))
        splits_ctpt = list(kf.split(np.arange(n_trials_ctpt)))

        # Iterate through both splits using zip
        for i_fold, (
            (trial_indices_self_train, trial_indices_self_test), 
            (trial_indices_ctpt_train, trial_indices_ctpt_test)) in enumerate(zip(splits_self, splits_ctpt)):
            
            print('fold: ', i_fold)

            if data_format in ['fill0', 'truncate_front', 'truncate_end', 'resample']:
                X_train_self = firing_rates_self_simple[trial_indices_self_train, :, :]
                X_test_ctpt  = firing_rates_ctpt_simple[trial_indices_ctpt_test, :, :]

                if input_firing_rates_self_simple is not None:
                    X_input_train_self = input_firing_rates_self_simple[trial_indices_self_train, :, :] 
                    X_input_test_ctpt  = input_firing_rates_ctpt_simple[trial_indices_ctpt_test, :, :]
                else:
                    X_input_train_self = None
                    X_input_test_ctpt  = None

                trial_lengths_train_self = trial_lengths_self[trial_indices_self_train]
                trial_lengths_test_ctpt  = trial_lengths_ctpt[trial_indices_ctpt_test]
            else:
                X_train_self = [firing_rates_self_simple[i] for i in trial_indices_self_train]
                X_test_ctpt  = [firing_rates_ctpt_simple[i] for i in trial_indices_ctpt_test]

                if input_firing_rates_self_simple is not None:
                    X_input_train_self = [input_firing_rates_self_simple[i] for i in trial_indices_self_train]
                    X_input_test_ctpt  = [input_firing_rates_ctpt_simple[i] for i in trial_indices_ctpt_test]
                else:
                    X_input_train_self = None
                    X_input_test_ctpt  = None

                trial_lengths_train_self = None
                trial_lengths_test_ctpt  = None
            
            ## Select data based on trial filter
            X_train = X_train_self
            X_test  = X_test_ctpt
            X_input_train = X_input_train_self
            X_input_test  = X_input_test_ctpt
            trial_lengths_train = trial_lengths_train_self
            trial_lengths_test  = trial_lengths_test_ctpt

            ## Sweep through candidate state/iteration counts for the fold
            for i_continuous_states, n_continuous_states in enumerate(ns_states):
                for i_discrete_states, n_discrete_states in enumerate(ns_discrete_states):
                    for i_iters, n_iters in enumerate(ns_iters):

                        time_start = time.time()

                        ## Format save path that encodes hyper-parameters and fold index
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
    #             'n_trials'      : n_trials_self,
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
        print('Running Cross Speed SLDS')
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
