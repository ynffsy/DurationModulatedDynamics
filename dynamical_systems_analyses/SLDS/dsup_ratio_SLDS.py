import os
import time
import ipdb
import tqdm
import pickle
import itertools
import numpy as np

from sklearn.model_selection import KFold

import dynamical_systems_analyses.SLDS.config as config
import dynamical_systems_analyses.utils.utils_processing as utils_processing
import dynamical_systems_analyses.utils.utils_dsup as utils_dsup
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
    alpha):    

    session_results_dir = os.path.join(results_dir, session_data_name)

    ## Initialize save name
    res_save_name = '_'.join(map(str, [x for x in [
        'dsupr',
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

    dsupr_fast = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_discrete_states), len(ns_iters)))
    dsupr_slow = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_discrete_states), len(ns_iters)))

    ## Number of trials that have zero length after filtering
    n_trials_fast_zero = np.sum(trial_lengths_fast == 0)
    n_trials_slow_zero = np.sum(trial_lengths_slow == 0)

    ## DSUP ratios per time point and trial (Order is preserved)
    trial_length_fast_max = np.max(trial_lengths_fast)
    trial_length_slow_max = np.max(trial_lengths_slow)

    dsupr_fast_test_per_time = np.full((len(random_states), len(ns_states), len(ns_discrete_states), len(ns_iters), n_trials_fast, trial_length_fast_max - 1), np.nan, dtype=float)
    dsupr_slow_test_per_time = np.full((len(random_states), len(ns_states), len(ns_discrete_states), len(ns_iters), n_trials_slow, trial_length_slow_max - 1), np.nan, dtype=float)

    ## Read data and decode cursor states
    for i_rs, random_state in enumerate(random_states):

        ## Multi-fold cross validation
        np.random.seed(random_state)

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('random state: ', random_state)

        ## K-fold cross validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        splits_fast = list(kf.split(np.arange(n_trials_fast)))
        splits_slow = list(kf.split(np.arange(n_trials_slow)))

        for i_fold, (
            (trial_indices_slow_train, trial_indices_slow_test), 
            (trial_indices_fast_train, trial_indices_fast_test)) in enumerate(zip(splits_slow, splits_fast)):
            
            print('fold: ', i_fold)

            if data_format is None:
                X_fast_train = [firing_rates_fast_simple[i] for i in trial_indices_fast_train]
                X_fast_test  = [firing_rates_fast_simple[i] for i in trial_indices_fast_test]
                X_slow_train = [firing_rates_slow_simple[i] for i in trial_indices_slow_train]
                X_slow_test  = [firing_rates_slow_simple[i] for i in trial_indices_slow_test]
            
            else:
                X_fast_train = np.array(firing_rates_fast_simple[trial_indices_fast_train])
                X_fast_test  = np.array(firing_rates_fast_simple[trial_indices_fast_test])
                X_slow_train = np.array(firing_rates_slow_simple[trial_indices_slow_train])
                X_slow_test  = np.array(firing_rates_slow_simple[trial_indices_slow_test])

            ## Sweep through various numbers of states and iterations
            for i_continuous_states, n_continuous_states in enumerate(ns_states):
                for i_discrete_states, n_discrete_states in enumerate(ns_discrete_states):
                    for i_iters, n_iters in enumerate(ns_iters):

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

                        ## Read SLDS results
                        try:
                            with open(fast_res_save_path, 'rb') as f:
                                res_SLDS_fast = pickle.load(f)
                            with open(slow_res_save_path, 'rb') as f:
                                res_SLDS_slow = pickle.load(f)
                            
                        except:
                            print('Model results not found')
                            print('fast_res_save_path: ', fast_res_save_path)
                            print('slow_res_save_path: ', slow_res_save_path)
                            return

                        continuous_states_fast_train = res_SLDS_fast['train_continuous_states']
                        continuous_states_fast_test  = res_SLDS_fast['test_continuous_states']
                        continuous_states_slow_train = res_SLDS_slow['train_continuous_states']
                        continuous_states_slow_test  = res_SLDS_slow['test_continuous_states']

                        continuous_states_fast_train_cov = res_SLDS_fast['train_continuous_state_covariances']
                        continuous_states_fast_test_cov  = res_SLDS_fast['test_continuous_state_covariances']
                        continuous_states_slow_train_cov = res_SLDS_slow['train_continuous_state_covariances']
                        continuous_states_slow_test_cov  = res_SLDS_slow['test_continuous_state_covariances']

                        discrete_states_fast_train = res_SLDS_fast['train_discrete_states']
                        discrete_states_fast_test  = res_SLDS_fast['test_discrete_states']
                        discrete_states_slow_train = res_SLDS_slow['train_discrete_states']
                        discrete_states_slow_test  = res_SLDS_slow['test_discrete_states']

                        model_fast = res_SLDS_fast['model']
                        model_slow = res_SLDS_slow['model']

                        dynamics_fast = model_fast.dynamics
                        dynamics_slow = model_slow.dynamics
                        
                        emissions_fast = model_fast.emissions
                        emissions_slow = model_slow.emissions

                        obs_mat_fast = emissions_fast.Cs[0]
                        obs_mat_slow = emissions_slow.Cs[0]
                        obs_bias_fast = emissions_fast.ds[0]
                        obs_bias_slow = emissions_slow.ds[0]

                        # obs_cov_fast = np.diag(1.0 / emissions_fast.inv_etas[0])
                        # obs_cov_slow = np.diag(1.0 / emissions_slow.inv_etas[0])
                        obs_cov_fast = np.diag(emissions_fast.inv_etas[0])
                        obs_cov_slow = np.diag(emissions_slow.inv_etas[0])
                        
                        dynamics_mats_fast = dynamics_fast.As
                        dynamics_mats_slow = dynamics_slow.As
                        dynamics_biases_fast = dynamics_fast.bs
                        dynamics_biases_slow = dynamics_slow.bs
                        dynamics_covs_fast = dynamics_fast.Sigmas
                        dynamics_covs_slow = dynamics_slow.Sigmas

                        X_fast_latent_fast_train = continuous_states_fast_train ## Fast latent space fast train data
                        X_slow_latent_slow_train = continuous_states_slow_train ## Slow latent space slow train data
                        assert len(X_fast_train) == len(X_fast_latent_fast_train)
                        assert len(X_slow_train) == len(X_slow_latent_slow_train)
                        X_fast_latent_fast_train_cov = continuous_states_fast_train_cov
                        X_slow_latent_slow_train_cov = continuous_states_slow_train_cov

                        if train_test_option in ['same_speed', 'joint']:
                            X_fast_latent_fast_test = continuous_states_fast_test  ## Fast latent space fast test data
                            X_slow_latent_slow_test = continuous_states_slow_test  ## Slow latent space slow test data
                            assert len(X_fast_test) == len(X_fast_latent_fast_test)
                            assert len(X_slow_test) == len(X_slow_latent_slow_test)
                            X_fast_latent_fast_test_cov = continuous_states_fast_test_cov
                            X_slow_latent_slow_test_cov = continuous_states_slow_test_cov
                            
                        else:
                            X_fast_latent_slow_test = continuous_states_fast_test  ## Fast latent space slow test data
                            X_slow_latent_fast_test = continuous_states_slow_test  ## Slow latent space fast test data
                            assert len(X_fast_test) == len(X_slow_latent_fast_test)
                            assert len(X_slow_test) == len(X_fast_latent_slow_test)
                            X_fast_latent_slow_test_cov = continuous_states_fast_test_cov
                            X_slow_latent_fast_test_cov = continuous_states_slow_test_cov

                        ## same-speed DSUP ratio
                        if train_test_option in ['same_speed', 'joint']:
                            fast_latent_train_dsupr, _ = utils_dsup.dsup_ratio_SLDS(
                                X_fast_train,
                                X_fast_latent_fast_train,
                                X_fast_latent_fast_train_cov,
                                obs_mat_fast,
                                obs_bias_fast,
                                obs_cov_fast,
                                dynamics_mats_fast,
                                dynamics_biases_fast,
                                dynamics_covs_fast,
                                discrete_states_fast_train)
                            
                            slow_latent_train_dsupr, _ = utils_dsup.dsup_ratio_SLDS(
                                X_slow_train, 
                                X_slow_latent_slow_train, 
                                X_slow_latent_slow_train_cov, 
                                obs_mat_slow,
                                obs_bias_slow,
                                obs_cov_slow,
                                dynamics_mats_slow,
                                dynamics_biases_slow,
                                dynamics_covs_slow,
                                discrete_states_slow_train)

                            fast_latent_test_dsupr, fast_latent_test_dsupr_per_time = utils_dsup.dsup_ratio_SLDS(
                                X_fast_test,
                                X_fast_latent_fast_test,
                                X_fast_latent_fast_test_cov,
                                obs_mat_fast,
                                obs_bias_fast,
                                obs_cov_fast,
                                dynamics_mats_fast,
                                dynamics_biases_fast,
                                dynamics_covs_fast,
                                discrete_states_fast_test)
                            
                            slow_latent_test_dsupr, slow_latent_test_dsupr_per_time = utils_dsup.dsup_ratio_SLDS(
                                X_slow_test,
                                X_slow_latent_slow_test,
                                X_slow_latent_slow_test_cov,
                                obs_mat_slow,
                                obs_bias_slow,
                                obs_cov_slow,
                                dynamics_mats_slow,
                                dynamics_biases_slow,
                                dynamics_covs_slow,
                                discrete_states_slow_test)
                            
                            ## Save per-time DSUP ratio
                            cumulative_trial_length_fast = 0
                            cumulative_trial_length_slow = 0

                            for trial_id in trial_indices_fast_test:
                                trial_length = trial_lengths_fast[trial_id] - 1
                                dsupr_fast_test_per_time[
                                    i_rs, 
                                    i_continuous_states, 
                                    i_discrete_states, 
                                    i_iters, 
                                    trial_id,
                                    :trial_length] = fast_latent_test_dsupr_per_time[cumulative_trial_length_fast : cumulative_trial_length_fast + trial_length]
                                
                                cumulative_trial_length_fast += trial_length

                            for trial_id in trial_indices_slow_test:
                                trial_length = trial_lengths_slow[trial_id] - 1
                                dsupr_slow_test_per_time[
                                    i_rs, 
                                    i_continuous_states, 
                                    i_discrete_states, 
                                    i_iters, 
                                    trial_id,
                                    :trial_length] = slow_latent_test_dsupr_per_time[cumulative_trial_length_slow : cumulative_trial_length_slow + trial_length]
                                
                                cumulative_trial_length_slow += trial_length

                        ## cross-speed DSUP ratio
                        else:
                            fast_latent_train_dsupr, _ = utils_dsup.dsup_ratio_SLDS(
                                X_fast_train,
                                X_fast_latent_fast_train,
                                X_fast_latent_fast_train_cov,
                                obs_mat_fast,
                                obs_bias_fast,
                                obs_cov_fast,
                                dynamics_mats_fast,
                                dynamics_biases_fast,
                                dynamics_covs_fast,
                                discrete_states_fast_train)
                            
                            slow_latent_train_dsupr, _ = utils_dsup.dsup_ratio_SLDS(
                                X_slow_train,
                                X_slow_latent_slow_train,
                                X_slow_latent_slow_train_cov,
                                obs_mat_slow,
                                obs_bias_slow,
                                obs_cov_slow,
                                dynamics_mats_slow,
                                dynamics_biases_slow,
                                dynamics_covs_slow,
                                discrete_states_slow_train)

                            fast_latent_test_dsupr, fast_latent_test_dsupr_per_time = utils_dsup.dsup_ratio_SLDS(
                                X_slow_test,
                                X_fast_latent_slow_test,
                                X_fast_latent_slow_test_cov,
                                obs_mat_fast,
                                obs_bias_fast,
                                obs_cov_fast,
                                dynamics_mats_fast,
                                dynamics_biases_fast,
                                dynamics_covs_fast,
                                discrete_states_fast_test)

                            slow_latent_test_dsupr, slow_latent_test_dsupr_per_time = utils_dsup.dsup_ratio_SLDS(
                                X_fast_test,
                                X_slow_latent_fast_test,
                                X_slow_latent_fast_test_cov,
                                obs_mat_slow,
                                obs_bias_slow,
                                obs_cov_slow,
                                dynamics_mats_slow,
                                dynamics_biases_slow,
                                dynamics_covs_slow,
                                discrete_states_slow_test)
                            
                            ## Save per-time DSUP ratio
                            cumulative_trial_length_slow = 0
                            cumulative_trial_length_fast = 0

                            for trial_id in trial_indices_slow_test:
                                trial_length = trial_lengths_slow[trial_id] - 1
                                dsupr_slow_test_per_time[
                                    i_rs, 
                                    i_continuous_states, 
                                    i_discrete_states, 
                                    i_iters, 
                                    trial_id,
                                    :trial_length] = fast_latent_test_dsupr_per_time[cumulative_trial_length_slow : cumulative_trial_length_slow + trial_length]
                                
                                cumulative_trial_length_slow += trial_length

                            for trial_id in trial_indices_fast_test:
                                trial_length = trial_lengths_fast[trial_id] - 1
                                dsupr_fast_test_per_time[
                                    i_rs, 
                                    i_continuous_states, 
                                    i_discrete_states, 
                                    i_iters, 
                                    trial_id,
                                    :trial_length] = slow_latent_test_dsupr_per_time[cumulative_trial_length_fast : cumulative_trial_length_fast + trial_length]
                                
                                cumulative_trial_length_fast += trial_length

                        ## Save results
                        dsupr_fast[0, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = fast_latent_train_dsupr
                        dsupr_fast[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = fast_latent_test_dsupr
                        dsupr_slow[0, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = slow_latent_train_dsupr
                        dsupr_slow[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = slow_latent_test_dsupr                        

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    ## Save results
    np.savez(res_save_path, 
        dsupr_fast=dsupr_fast,
        dsupr_slow=dsupr_slow,
        dsupr_fast_test_per_time=dsupr_fast_test_per_time,
        dsupr_slow_test_per_time=dsupr_slow_test_per_time,
        trial_lengths_fast=trial_lengths_fast,
        trial_lengths_slow=trial_lengths_slow)



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
        print('Calculating DSUP Ratio ...')
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
    