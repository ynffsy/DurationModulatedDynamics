import os
import time
import ipdb
import pickle
import itertools

import numpy as np

from sklearn.model_selection import KFold

import dynamical_systems_analyses.SLDS.config as config
import dynamical_systems_analyses.utils.utils_processing as utils_processing
from SLDS import SLDS
from vis_config import session_target_radii

        

## Read parameters from config
overwrite_results  = config.overwrite_results
data_dir           = config.data_dir
results_dir        = config.results_dir
vis_dir            = config.vis_dir
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
    model_type,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha):

    assert len(trial_filters) == 2
    assert data_format is None

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

    ## Get paired data for self and ctpt (counterpart) trials
    (firing_rates_self_simple, firing_rates_ctpt_simple, 
     input_firing_rates_self_simple, input_firing_rates_ctpt_simple,
     _, _, 
     n_trials_self, n_trials_ctpt,
     _, _,
     _, _,
     _, _) = data_loader.reformat_firing_rate_data(data_format)

    n_neurons = firing_rates_self_simple[0].shape[1]
    assert n_neurons == firing_rates_ctpt_simple[0].shape[1]

    if input_firing_rates_self_simple is not None:
        n_input_neurons = input_firing_rates_self_simple[0].shape[1]
        assert n_input_neurons == input_firing_rates_ctpt_simple[0].shape[1]
    else:
        n_input_neurons = 0
    
    model_results_dir_self, model_results_dir_ctpt = data_loader.get_model_result_dirs(
        time_offset=time_offset,
        data_format=data_format,
        train_test='joint',
        model_type=model_type,
        dynamics_class=dynamics_class,
        emission_class=emission_class,
        init_type=init_type,
        subspace_type=subspace_type,
        alpha=alpha,
        check_existence=False)


    for i_rs, random_state in enumerate(random_states):

        np.random.seed(random_state)

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('random state: ', random_state)

        ## K-fold cross validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        splits_self = list(kf.split(np.arange(n_trials_self)))
        splits_ctpt = list(kf.split(np.arange(n_trials_ctpt)))

        # Iterate through both splits using zip
        for i_fold, (
            (trial_indices_self_train, trial_indices_self_test), 
            (trial_indices_ctpt_train, trial_indices_ctpt_test)) in enumerate(zip(splits_self, splits_ctpt)):

            print('fold: ', i_fold)

            X_train_self = [firing_rates_self_simple[i] for i in trial_indices_self_train]
            X_train_ctpt = [firing_rates_ctpt_simple[i] for i in trial_indices_ctpt_train]
            X_test_ctpt  = [firing_rates_ctpt_simple[i] for i in trial_indices_ctpt_test]
            X_test_self  = [firing_rates_self_simple[i] for i in trial_indices_self_test]

            if input_firing_rates_self_simple is not None:
                X_input_train_self = [input_firing_rates_self_simple[i] for i in trial_indices_self_train]
                X_input_train_ctpt = [input_firing_rates_ctpt_simple[i] for i in trial_indices_ctpt_train]
                X_input_test_ctpt  = [input_firing_rates_ctpt_simple[i] for i in trial_indices_ctpt_test]
                X_input_test_self  = [input_firing_rates_self_simple[i] for i in trial_indices_self_test]
            else:
                X_input_train_self = None
                X_input_train_ctpt = None
                X_input_test_ctpt  = None
                X_input_test_self  = None

            ## Get shape of data
            n_trials_X_train_self = len(X_train_self)
            n_trials_X_train_ctpt = len(X_train_ctpt)
            n_trials_X_test_self  = len(X_test_self)
            n_trials_X_test_ctpt  = len(X_test_ctpt)

            ## Combine self and ctpt data
            X_train = X_train_self + X_train_ctpt
            X_test  = X_test_self + X_test_ctpt

            if input_firing_rates_self_simple is not None:
                X_input_train = X_input_train_self + X_input_train_ctpt
                X_input_test  = X_input_test_self + X_input_test_ctpt
            else:
                X_input_train = None
                X_input_test  = None

            ## Sweep through various numbers of states and iterations as well as random states
            for i_continuous_states, n_continuous_states in enumerate(ns_states):
                for i_discrete_states, n_discrete_states in enumerate(ns_discrete_states):
                    for i_iters, n_iters in enumerate(ns_iters):

                        time_start = time.time()

                        ## Format save path
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

                        model_save_prefix_self = os.path.join(model_results_dir_self, model_save_name)
                        model_save_prefix_ctpt = os.path.join(model_results_dir_ctpt, model_save_name)
                        res_save_path_self     = model_save_prefix_self + '.pkl'
                        res_save_path_ctpt     = model_save_prefix_ctpt + '.pkl'

                        if not overwrite_results:
                            if os.path.isfile(res_save_path_self) and os.path.isfile(res_save_path_ctpt):
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

                        train_continuous_states_self = neural_SLDS.train_continuous_states[:n_trials_X_train_self]
                        train_continuous_states_ctpt = neural_SLDS.train_continuous_states[n_trials_X_train_self:n_trials_X_train_self + n_trials_X_train_ctpt]
                        train_continuous_state_covariances_self = neural_SLDS.train_continuous_state_covariances[:n_trials_X_train_self]
                        train_continuous_state_covariances_ctpt = neural_SLDS.train_continuous_state_covariances[n_trials_X_train_self:n_trials_X_train_self + n_trials_X_train_ctpt]
                        train_discrete_states_self = neural_SLDS.train_discrete_states[:n_trials_X_train_self]
                        train_discrete_states_ctpt = neural_SLDS.train_discrete_states[n_trials_X_train_self:n_trials_X_train_self + n_trials_X_train_ctpt]
                        test_continuous_states_self = neural_SLDS.test_continuous_states[:n_trials_X_test_self]
                        test_continuous_states_ctpt = neural_SLDS.test_continuous_states[n_trials_X_test_self:n_trials_X_test_self + n_trials_X_test_ctpt]
                        test_continuous_state_covariances_self = neural_SLDS.test_continuous_state_covariances[:n_trials_X_test_self]
                        test_continuous_state_covariances_ctpt = neural_SLDS.test_continuous_state_covariances[n_trials_X_test_self:n_trials_X_test_self + n_trials_X_test_ctpt]
                        test_discrete_states_self = neural_SLDS.test_discrete_states[:n_trials_X_test_self]
                        test_discrete_states_ctpt = neural_SLDS.test_discrete_states[n_trials_X_test_self:n_trials_X_test_self + n_trials_X_test_ctpt]

                        with open(res_save_path_self, 'wb') as f:
                            pickle.dump({
                                'train_elbos'                        : neural_SLDS.train_elbos,
                                'train_continuous_states'            : train_continuous_states_self,
                                'train_continuous_state_covariances' : train_continuous_state_covariances_self,
                                'train_discrete_states'              : train_discrete_states_self,
                                'test_elbos'                         : neural_SLDS.test_elbos,
                                'test_continuous_states'             : test_continuous_states_self,
                                'test_continuous_state_covariances'  : test_continuous_state_covariances_self,
                                'test_discrete_states'               : test_discrete_states_self,
                                'model'                              : neural_SLDS.model,
                            }, f)

                        with open(res_save_path_ctpt, 'wb') as f:
                            pickle.dump({
                                'train_elbos'                        : neural_SLDS.train_elbos,
                                'train_continuous_states'            : train_continuous_states_ctpt,
                                'train_continuous_state_covariances' : train_continuous_state_covariances_ctpt,
                                'train_discrete_states'              : train_discrete_states_ctpt,
                                'test_elbos'                         : neural_SLDS.test_elbos,
                                'test_continuous_states'             : test_continuous_states_ctpt,
                                'test_continuous_state_covariances'  : test_continuous_state_covariances_ctpt,
                                'test_discrete_states'               : test_discrete_states_ctpt,
                                'model'                              : neural_SLDS.model,
                            }, f)

                        time_end = time.time()
                        runtime = time_end - time_start

                        print('# continuous states: ', n_continuous_states, ' # discrete states: ', n_discrete_states, ' # iters: ', n_iters, ' run time: ', runtime)

            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')



if __name__ == '__main__':

    for (
        session_data_name, 
        unit_filter, 
        input_unit_filter, 
        data_format,
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
            model_types, 
            dynamics_classes, 
            emission_classes,
            init_types, 
            subspace_types,
            alphas):

        print('=============================================================')
        print('Running joint SLDS')
        print('\tsession_data_name: ',    session_data_name)
        print('\tunit_filter: ',          unit_filter)
        print('\tinput_unit_filter: ',    input_unit_filter)
        print('\twindow_config: ',        window_config)
        print('\tdata_format: ',          data_format)
        print('\ttrial_filters: ',        trial_filters)
        print('\trandom_states: ',        random_states)
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
            model_type, 
            dynamics_class,
            emission_class,
            init_type, 
            subspace_type,
            alpha)
