"""Run SLDS-based inference benchmarks (dynamics, emissions, forecasts)."""

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
import dynamical_systems_analyses.utils.utils_decoding as utils_decoding
import dynamical_systems_analyses.utils.utils_inference as utils_inference
from vis_config import session_target_radii



# -----------------------------------------------------------------------------
# Shared configuration for exhaustive inference sweeps
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
    alpha):
    """
    3 types of inferences:
    dynamics_inference: use the dynamics matrices to learn the latent state at time t+1 from the latent state at time t 
    emissions_inference: use the emissions matrices to learn the latent state at time t from the emissions at time t
    forecast_inference: use the dynamics matrices to learn the latent state at time t+1 from the latent state at time t, 
        then use the emissions matrices to learn the emissions at time t+1 from the inferred latent state at time t+1
    """

    session_results_dir = os.path.join(results_dir, session_data_name)

    ## Initialize save name
    res_save_name = '_'.join(map(str, [x for x in [
        'inference',
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

    ## Load spike trains + cursor data for the requested preprocessing
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
     input_firing_rates_fast_simple, input_firing_rates_slow_simple, 
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

    ## Initialize results
    ## (train/test axis first, followed by random_state -> fold -> hyper-params)
    ## Cannot save inference results for training data because results from each fold would overwrite each other
    dynamics_inference_fast_test = np.empty((len(random_states), len(ns_states), len(ns_discrete_states), len(ns_iters), n_trials_fast), dtype=object)
    dynamics_inference_slow_test = np.empty((len(random_states), len(ns_states), len(ns_discrete_states), len(ns_iters), n_trials_slow), dtype=object)

    emissions_inference_fast_test = np.empty((len(random_states), len(ns_states), len(ns_discrete_states), len(ns_iters), n_trials_fast), dtype=object)
    emissions_inference_slow_test = np.empty((len(random_states), len(ns_states), len(ns_discrete_states), len(ns_iters), n_trials_slow), dtype=object)

    forecast_fast_test = np.empty((len(random_states), len(ns_states), len(ns_discrete_states), len(ns_iters), n_trials_fast), dtype=object)
    forecast_slow_test = np.empty((len(random_states), len(ns_states), len(ns_discrete_states), len(ns_iters), n_trials_slow), dtype=object)

    ## The 1st dimension has 2 elements: train and test
    rmse_dynamics_inference_fast = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_discrete_states), len(ns_iters)))
    rmse_dynamics_inference_slow = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_discrete_states), len(ns_iters)))
    r2_dynamics_inference_fast   = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_discrete_states), len(ns_iters)))
    r2_dynamics_inference_slow   = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_discrete_states), len(ns_iters)))

    rmse_emissions_inference_fast = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_discrete_states), len(ns_iters)))
    rmse_emissions_inference_slow = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_discrete_states), len(ns_iters)))
    r2_emissions_inference_fast   = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_discrete_states), len(ns_iters)))
    r2_emissions_inference_slow   = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_discrete_states), len(ns_iters)))

    rmse_forecast_fast = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_discrete_states), len(ns_iters)))
    rmse_forecast_slow = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_discrete_states), len(ns_iters)))
    r2_forecast_fast   = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_discrete_states), len(ns_iters)))
    r2_forecast_slow   = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_discrete_states), len(ns_iters)))

    ## Inference metrics per time point and trial (order preserved; R2 computed per sample)
    trial_length_fast_max = np.max(trial_lengths_fast)
    trial_length_slow_max = np.max(trial_lengths_slow)

    r2_dynamics_inference_fast_test_per_time = np.full((len(random_states), len(ns_states), len(ns_discrete_states), len(ns_iters), n_trials_fast, trial_length_fast_max - 1), np.nan, dtype=float)
    r2_dynamics_inference_slow_test_per_time = np.full((len(random_states), len(ns_states), len(ns_discrete_states), len(ns_iters), n_trials_slow, trial_length_slow_max - 1), np.nan, dtype=float)

    r2_emissions_inference_fast_test_per_time = np.full((len(random_states), len(ns_states), len(ns_discrete_states), len(ns_iters), n_trials_fast, trial_length_fast_max), np.nan, dtype=float)
    r2_emissions_inference_slow_test_per_time = np.full((len(random_states), len(ns_states), len(ns_discrete_states), len(ns_iters), n_trials_slow, trial_length_slow_max), np.nan, dtype=float)

    r2_forecast_fast_test_per_time = np.full((len(random_states), len(ns_states), len(ns_discrete_states), len(ns_iters), n_trials_fast, trial_length_fast_max - 1), np.nan, dtype=float)
    r2_forecast_slow_test_per_time = np.full((len(random_states), len(ns_states), len(ns_discrete_states), len(ns_iters), n_trials_slow, trial_length_slow_max - 1), np.nan, dtype=float)


    ## Read data and decode cursor states
    for i_rs, random_state in enumerate(random_states):

        ## Multi-fold cross validation (repeatable due to fixed rng seed)
        np.random.seed(random_state)

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('random state: ', random_state)

        ## K-fold cross validation that mirrors the decoding scripts
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        splits_fast = list(kf.split(np.arange(n_trials_fast)))
        splits_slow = list(kf.split(np.arange(n_trials_slow)))

        # Iterate over paired (slow, fast) folds so the comparisons stay aligned
        for i_fold, (
            (trial_indices_slow_train, trial_indices_slow_test), 
            (trial_indices_fast_train, trial_indices_fast_test)) in enumerate(zip(splits_slow, splits_fast)):
            
            print('fold: ', i_fold)

            if data_format is None:
                X_fast_train = [firing_rates_fast_simple[i] for i in trial_indices_fast_train]
                X_fast_test  = [firing_rates_fast_simple[i] for i in trial_indices_fast_test]
                X_slow_train = [firing_rates_slow_simple[i] for i in trial_indices_slow_train]
                X_slow_test  = [firing_rates_slow_simple[i] for i in trial_indices_slow_test]

                X_fast_train_input = [input_firing_rates_fast_simple[i] for i in trial_indices_fast_train] if input_firing_rates_fast_simple is not None else None
                X_fast_test_input  = [input_firing_rates_fast_simple[i] for i in trial_indices_fast_test] if input_firing_rates_fast_simple is not None else None
                X_slow_train_input = [input_firing_rates_slow_simple[i] for i in trial_indices_slow_train] if input_firing_rates_slow_simple is not None else None
                X_slow_test_input  = [input_firing_rates_slow_simple[i] for i in trial_indices_slow_test] if input_firing_rates_slow_simple is not None else None
            
            else:
                X_fast_train = np.array(firing_rates_fast_simple[trial_indices_fast_train])
                X_fast_test  = np.array(firing_rates_fast_simple[trial_indices_fast_test])
                X_slow_train = np.array(firing_rates_slow_simple[trial_indices_slow_train])
                X_slow_test  = np.array(firing_rates_slow_simple[trial_indices_slow_test])

                X_fast_train_input = np.array(input_firing_rates_fast_simple[trial_indices_fast_train]) if input_firing_rates_fast_simple is not None else None
                X_fast_test_input  = np.array(input_firing_rates_fast_simple[trial_indices_fast_test]) if input_firing_rates_fast_simple is not None else None
                X_slow_train_input = np.array(input_firing_rates_slow_simple[trial_indices_slow_train]) if input_firing_rates_slow_simple is not None else None
                X_slow_test_input  = np.array(input_firing_rates_slow_simple[trial_indices_slow_test]) if input_firing_rates_slow_simple is not None else None

            ## Concatenate trials
            X_fast_train_concat = np.concatenate(X_fast_train, axis=0)
            X_fast_test_concat  = np.concatenate(X_fast_test,  axis=0)
            X_slow_train_concat = np.concatenate(X_slow_train, axis=0)
            X_slow_test_concat  = np.concatenate(X_slow_test,  axis=0)

            ## Concatenate trials while removing first time point
            ## (needed so one-step-ahead forecasts align with available history)
            X_fast_train_no_first_concat = np.concatenate([X_fast_train[i][1:, :] for i in range(len(X_fast_train))], axis=0)
            X_fast_test_no_first_concat  = np.concatenate([X_fast_test[i][1:, :] for i in range(len(X_fast_test))], axis=0)
            X_slow_train_no_first_concat = np.concatenate([X_slow_train[i][1:, :] for i in range(len(X_slow_train))], axis=0)
            X_slow_test_no_first_concat  = np.concatenate([X_slow_test[i][1:, :] for i in range(len(X_slow_test))], axis=0)

            X_fast_train_trial_lengths = trial_lengths_fast[trial_indices_fast_train]
            X_fast_test_trial_lengths  = trial_lengths_fast[trial_indices_fast_test]
            X_slow_train_trial_lengths = trial_lengths_slow[trial_indices_slow_train]
            X_slow_test_trial_lengths  = trial_lengths_slow[trial_indices_slow_test]

            ## For SSM, sweep through various numbers of states and iterations
            for i_continuous_states, n_continuous_states in enumerate(ns_states):
                for i_discrete_states, n_discrete_states in enumerate(ns_discrete_states):
                    for i_iters, n_iters in enumerate(ns_iters):

                        print('n_continuous_states: ', n_continuous_states, ' n_discrete_states: ', n_discrete_states, ' n_iters: ', n_iters)

                        ## Read SLDS processed data (latent/posterior caches per fold/state)
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

                        # Latent trajectories for each train trial (kept separate for reshaping later)
                        X_fast_latent_fast_train = continuous_states_fast_train ## Fast latent space fast train data
                        X_slow_latent_slow_train = continuous_states_slow_train ## Slow latent space slow train data
                        
                        assert len(X_fast_train) == len(X_fast_latent_fast_train)
                        assert len(X_slow_train) == len(X_slow_latent_slow_train)

                        X_fast_latent_fast_train_concat = np.concatenate(X_fast_latent_fast_train, axis=0)
                        X_slow_latent_slow_train_concat = np.concatenate(X_slow_latent_slow_train, axis=0)
                        X_fast_latent_fast_train_no_first_concat = np.concatenate([X_fast_latent_fast_train[i][1:, :] for i in range(len(X_fast_latent_fast_train))], axis=0)
                        X_slow_latent_slow_train_no_first_concat = np.concatenate([X_slow_latent_slow_train[i][1:, :] for i in range(len(X_slow_latent_slow_train))], axis=0)

                        if train_test_option in ['same_speed', 'joint']:
                            X_fast_latent_fast_test  = continuous_states_fast_test  ## Fast latent space fast test data
                            X_slow_latent_slow_test  = continuous_states_slow_test  ## Slow latent space slow test data
                            assert len(X_fast_test)  == len(X_fast_latent_fast_test)
                            assert len(X_slow_test)  == len(X_slow_latent_slow_test)

                            X_fast_latent_fast_test_concat  = np.concatenate(X_fast_latent_fast_test, axis=0)
                            X_slow_latent_slow_test_concat  = np.concatenate(X_slow_latent_slow_test, axis=0)
                            X_fast_latent_fast_test_no_first_concat = np.concatenate([X_fast_latent_fast_test[i][1:, :] for i in range(len(X_fast_latent_fast_test))], axis=0)
                            X_slow_latent_slow_test_no_first_concat = np.concatenate([X_slow_latent_slow_test[i][1:, :] for i in range(len(X_slow_latent_slow_test))], axis=0)

                        else:
                            X_fast_latent_slow_test  = continuous_states_fast_test  ## Fast latent space slow test data
                            X_slow_latent_fast_test  = continuous_states_slow_test  ## Slow latent space fast test data
                            assert len(X_fast_test)  == len(X_slow_latent_fast_test)
                            assert len(X_slow_test)  == len(X_fast_latent_slow_test)

                            X_slow_latent_fast_test_concat  = np.concatenate(X_slow_latent_fast_test,  axis=0)
                            X_fast_latent_slow_test_concat  = np.concatenate(X_fast_latent_slow_test,  axis=0)
                            X_slow_latent_fast_test_no_first_concat = np.concatenate([X_slow_latent_fast_test[i][1:, :] for i in range(len(X_slow_latent_fast_test))], axis=0)
                            X_fast_latent_slow_test_no_first_concat = np.concatenate([X_fast_latent_slow_test[i][1:, :] for i in range(len(X_fast_latent_slow_test))], axis=0)


                        #### Inference ####
                        X_forecast_fast_train = utils_inference.forecast_inference_SLDS(X_fast_latent_fast_train, discrete_states_fast_train, X_fast_train_trial_lengths, dynamics_fast, emissions_fast, X_fast_train_input)
                        X_forecast_slow_train = utils_inference.forecast_inference_SLDS(X_slow_latent_slow_train, discrete_states_slow_train, X_slow_train_trial_lengths, dynamics_slow, emissions_slow, X_slow_train_input)

                        X_dynamics_inference_fast_train = utils_inference.dynamics_inference_SLDS(X_fast_latent_fast_train, discrete_states_fast_train, X_fast_train_trial_lengths, dynamics_fast, X_fast_train_input)
                        X_dynamics_inference_slow_train = utils_inference.dynamics_inference_SLDS(X_slow_latent_slow_train, discrete_states_slow_train, X_slow_train_trial_lengths, dynamics_slow, X_slow_train_input)
                        
                        X_emissions_inference_fast_train = utils_inference.emissions_inference_SLDS(X_fast_latent_fast_train, emissions_fast, X_fast_train_input)
                        X_emissions_inference_slow_train = utils_inference.emissions_inference_SLDS(X_slow_latent_slow_train, emissions_slow, X_slow_train_input)

                        ## Same-speed inference
                        if train_test_option in ['same_speed', 'joint']:
                            X_forecast_fast_test = utils_inference.forecast_inference_SLDS(X_fast_latent_fast_test, discrete_states_fast_test, X_fast_test_trial_lengths, dynamics_fast, emissions_fast, X_fast_test_input)
                            X_forecast_slow_test = utils_inference.forecast_inference_SLDS(X_slow_latent_slow_test, discrete_states_slow_test, X_slow_test_trial_lengths, dynamics_slow, emissions_slow, X_slow_test_input)

                            X_dynamics_inference_fast_test = utils_inference.dynamics_inference_SLDS(X_fast_latent_fast_test, discrete_states_fast_test, X_fast_test_trial_lengths, dynamics_fast, X_fast_test_input)
                            X_dynamics_inference_slow_test = utils_inference.dynamics_inference_SLDS(X_slow_latent_slow_test, discrete_states_slow_test, X_slow_test_trial_lengths, dynamics_slow, X_slow_test_input)

                            X_emissions_inference_fast_test = utils_inference.emissions_inference_SLDS(X_fast_latent_fast_test, emissions_fast, X_fast_test_input)
                            X_emissions_inference_slow_test = utils_inference.emissions_inference_SLDS(X_slow_latent_slow_test, emissions_slow, X_slow_test_input)

                        ## Cross-speed inference
                        else:
                            ## Train slow test fast
                            X_forecast_fast_test = utils_inference.forecast_inference_SLDS(X_slow_latent_fast_test, discrete_states_slow_test, X_fast_test_trial_lengths, dynamics_slow, emissions_slow, X_fast_test_input)
                            X_forecast_slow_test = utils_inference.forecast_inference_SLDS(X_fast_latent_slow_test, discrete_states_fast_test, X_slow_test_trial_lengths, dynamics_fast, emissions_fast, X_slow_test_input)

                            X_dynamics_inference_fast_test = utils_inference.dynamics_inference_SLDS(X_slow_latent_fast_test, discrete_states_slow_test, X_fast_test_trial_lengths, dynamics_slow, X_fast_test_input)
                            X_dynamics_inference_slow_test = utils_inference.dynamics_inference_SLDS(X_fast_latent_slow_test, discrete_states_fast_test, X_slow_test_trial_lengths, dynamics_fast, X_slow_test_input)

                            X_emissions_inference_fast_test = utils_inference.emissions_inference_SLDS(X_slow_latent_fast_test, emissions_slow, X_fast_test_input)
                            X_emissions_inference_slow_test = utils_inference.emissions_inference_SLDS(X_fast_latent_slow_test, emissions_fast, X_slow_test_input)


                        ## Concatenate inference results
                        X_forecast_fast_train_concat = np.concatenate(X_forecast_fast_train, axis=0)
                        X_forecast_fast_test_concat  = np.concatenate(X_forecast_fast_test,  axis=0)
                        X_forecast_slow_train_concat = np.concatenate(X_forecast_slow_train, axis=0)
                        X_forecast_slow_test_concat  = np.concatenate(X_forecast_slow_test,  axis=0)
                        
                        X_dynamics_inference_fast_train_concat = np.concatenate(X_dynamics_inference_fast_train, axis=0)
                        X_dynamics_inference_fast_test_concat  = np.concatenate(X_dynamics_inference_fast_test,  axis=0)
                        X_dynamics_inference_slow_train_concat = np.concatenate(X_dynamics_inference_slow_train, axis=0)
                        X_dynamics_inference_slow_test_concat  = np.concatenate(X_dynamics_inference_slow_test,  axis=0)
                        
                        X_emissions_inference_fast_train_concat = np.concatenate(X_emissions_inference_fast_train, axis=0)
                        X_emissions_inference_fast_test_concat  = np.concatenate(X_emissions_inference_fast_test,  axis=0)
                        X_emissions_inference_slow_train_concat = np.concatenate(X_emissions_inference_slow_train, axis=0)
                        X_emissions_inference_slow_test_concat  = np.concatenate(X_emissions_inference_slow_test,  axis=0)

                        ## Concatenate inference results while removing last time point
                        X_forecast_fast_train_no_last = [X_forecast_fast_train[i][:-1, :] for i in range(len(X_forecast_fast_train))]
                        X_forecast_fast_test_no_last  = [X_forecast_fast_test[i][:-1, :] for i in range(len(X_forecast_fast_test))]
                        X_forecast_slow_train_no_last = [X_forecast_slow_train[i][:-1, :] for i in range(len(X_forecast_slow_train))]
                        X_forecast_slow_test_no_last  = [X_forecast_slow_test[i][:-1, :] for i in range(len(X_forecast_slow_test))]

                        X_forecast_fast_train_no_last_concat = np.concatenate(X_forecast_fast_train_no_last, axis=0)
                        X_forecast_fast_test_no_last_concat  = np.concatenate(X_forecast_fast_test_no_last,  axis=0)
                        X_forecast_slow_train_no_last_concat = np.concatenate(X_forecast_slow_train_no_last, axis=0)
                        X_forecast_slow_test_no_last_concat  = np.concatenate(X_forecast_slow_test_no_last,  axis=0)

                        X_dynamics_inference_fast_train_no_last = [X_dynamics_inference_fast_train[i][:-1, :] for i in range(len(X_dynamics_inference_fast_train))]
                        X_dynamics_inference_fast_test_no_last  = [X_dynamics_inference_fast_test[i][:-1, :] for i in range(len(X_dynamics_inference_fast_test))]
                        X_dynamics_inference_slow_train_no_last = [X_dynamics_inference_slow_train[i][:-1, :] for i in range(len(X_dynamics_inference_slow_train))]
                        X_dynamics_inference_slow_test_no_last  = [X_dynamics_inference_slow_test[i][:-1, :] for i in range(len(X_dynamics_inference_slow_test))]

                        X_dynamics_inference_fast_train_no_last_concat = np.concatenate(X_dynamics_inference_fast_train_no_last, axis=0)
                        X_dynamics_inference_fast_test_no_last_concat  = np.concatenate(X_dynamics_inference_fast_test_no_last,  axis=0)
                        X_dynamics_inference_slow_train_no_last_concat = np.concatenate(X_dynamics_inference_slow_train_no_last, axis=0)
                        X_dynamics_inference_slow_test_no_last_concat  = np.concatenate(X_dynamics_inference_slow_test_no_last,  axis=0)
                        
                        ## Save inference and forecast results
                        forecast_fast_test[i_rs, i_continuous_states, i_discrete_states, i_iters, trial_indices_fast_test] = X_forecast_fast_test_no_last
                        forecast_slow_test[i_rs, i_continuous_states, i_discrete_states, i_iters, trial_indices_slow_test] = X_forecast_slow_test_no_last
                        
                        dynamics_inference_fast_test[i_rs, i_continuous_states, i_discrete_states, i_iters, trial_indices_fast_test] = X_dynamics_inference_fast_test_no_last
                        dynamics_inference_slow_test[i_rs, i_continuous_states, i_discrete_states, i_iters, trial_indices_slow_test] = X_dynamics_inference_slow_test_no_last
                        
                        emissions_inference_fast_test[i_rs, i_continuous_states, i_discrete_states, i_iters, trial_indices_fast_test] = X_emissions_inference_fast_test
                        emissions_inference_slow_test[i_rs, i_continuous_states, i_discrete_states, i_iters, trial_indices_slow_test] = X_emissions_inference_slow_test


                        #### Compute inference errors ####

                        ## NOTE: Inference slow means the data used is slow. 
                        ## The model used to process the data can be slow or fast 
                        ##   depending on whether train_test is cross_speed or not.

                        ## Compute forecast error
                        rmse_forecast_fast_train = utils_decoding.rmse(X_forecast_fast_train_no_last_concat, X_fast_train_no_first_concat)
                        rmse_forecast_fast_test  = utils_decoding.rmse(X_forecast_fast_test_no_last_concat,  X_fast_test_no_first_concat)
                        rmse_forecast_slow_train = utils_decoding.rmse(X_forecast_slow_train_no_last_concat, X_slow_train_no_first_concat)
                        rmse_forecast_slow_test  = utils_decoding.rmse(X_forecast_slow_test_no_last_concat,  X_slow_test_no_first_concat)
                        
                        r2_forecast_fast_train = utils_decoding.r2(X_forecast_fast_train_no_last_concat, X_fast_train_no_first_concat)
                        r2_forecast_fast_test  = utils_decoding.r2(X_forecast_fast_test_no_last_concat,  X_fast_test_no_first_concat)
                        r2_forecast_slow_train = utils_decoding.r2(X_forecast_slow_train_no_last_concat, X_slow_train_no_first_concat)
                        r2_forecast_slow_test  = utils_decoding.r2(X_forecast_slow_test_no_last_concat,  X_slow_test_no_first_concat)

                        ## Save per-time forecast error
                        r2_forecast_fast_test_per_time_ = utils_decoding.r2(X_forecast_fast_test_no_last_concat, X_fast_test_no_first_concat, axis=1)
                        r2_forecast_slow_test_per_time_ = utils_decoding.r2(X_forecast_slow_test_no_last_concat, X_slow_test_no_first_concat, axis=1)

                        cumulative_trial_length_fast = 0
                        cumulative_trial_length_slow = 0

                        for trial_id in trial_indices_fast_test:
                            trial_length = trial_lengths_fast[trial_id] - 1

                            r2_forecast_fast_test_per_time[
                                i_rs,
                                i_continuous_states,
                                i_discrete_states,
                                i_iters,
                                trial_id,
                                :trial_length] = r2_forecast_fast_test_per_time_[cumulative_trial_length_fast:cumulative_trial_length_fast + trial_length]
                            
                            cumulative_trial_length_fast += trial_length

                        for trial_id in trial_indices_slow_test:
                            trial_length = trial_lengths_slow[trial_id] - 1

                            r2_forecast_slow_test_per_time[
                                i_rs,
                                i_continuous_states,
                                i_discrete_states,
                                i_iters,
                                trial_id,
                                :trial_length] = r2_forecast_slow_test_per_time_[cumulative_trial_length_slow:cumulative_trial_length_slow + trial_length]
                            
                            cumulative_trial_length_slow += trial_length


                        ## Compute dynamics inference error
                        rmse_dynamics_inference_fast_train = utils_decoding.rmse(X_dynamics_inference_fast_train_no_last_concat, X_fast_latent_fast_train_no_first_concat)
                        rmse_dynamics_inference_slow_train = utils_decoding.rmse(X_dynamics_inference_slow_train_no_last_concat, X_slow_latent_slow_train_no_first_concat)
                        r2_dynamics_inference_fast_train = utils_decoding.r2(X_dynamics_inference_fast_train_no_last_concat, X_fast_latent_fast_train_no_first_concat)
                        r2_dynamics_inference_slow_train = utils_decoding.r2(X_dynamics_inference_slow_train_no_last_concat, X_slow_latent_slow_train_no_first_concat)

                        if train_test_option in ['same_speed', 'joint']:
                            rmse_dynamics_inference_fast_test = utils_decoding.rmse(X_dynamics_inference_fast_test_no_last_concat, X_fast_latent_fast_test_no_first_concat)
                            rmse_dynamics_inference_slow_test = utils_decoding.rmse(X_dynamics_inference_slow_test_no_last_concat, X_slow_latent_slow_test_no_first_concat)
                            r2_dynamics_inference_fast_test = utils_decoding.r2(X_dynamics_inference_fast_test_no_last_concat, X_fast_latent_fast_test_no_first_concat)
                            r2_dynamics_inference_slow_test = utils_decoding.r2(X_dynamics_inference_slow_test_no_last_concat, X_slow_latent_slow_test_no_first_concat)

                            r2_dynamics_inference_fast_test_per_time_ = utils_decoding.r2(X_dynamics_inference_fast_test_no_last_concat, X_fast_latent_fast_test_no_first_concat, axis=1)
                            r2_dynamics_inference_slow_test_per_time_ = utils_decoding.r2(X_dynamics_inference_slow_test_no_last_concat, X_slow_latent_slow_test_no_first_concat, axis=1)

                        else:
                            rmse_dynamics_inference_fast_test = utils_decoding.rmse(X_dynamics_inference_fast_test_no_last_concat, X_slow_latent_fast_test_no_first_concat)
                            rmse_dynamics_inference_slow_test = utils_decoding.rmse(X_dynamics_inference_slow_test_no_last_concat, X_fast_latent_slow_test_no_first_concat)
                            r2_dynamics_inference_fast_test = utils_decoding.r2(X_dynamics_inference_fast_test_no_last_concat, X_slow_latent_fast_test_no_first_concat)
                            r2_dynamics_inference_slow_test = utils_decoding.r2(X_dynamics_inference_slow_test_no_last_concat, X_fast_latent_slow_test_no_first_concat)

                            r2_dynamics_inference_fast_test_per_time_ = utils_decoding.r2(X_dynamics_inference_fast_test_no_last_concat, X_slow_latent_fast_test_no_first_concat, axis=1)
                            r2_dynamics_inference_slow_test_per_time_ = utils_decoding.r2(X_dynamics_inference_slow_test_no_last_concat, X_fast_latent_slow_test_no_first_concat, axis=1)

                        ## Save per-time forecast error
                        cumulative_trial_length_fast = 0
                        cumulative_trial_length_slow = 0

                        for trial_id in trial_indices_fast_test:
                            trial_length = trial_lengths_fast[trial_id] - 1

                            r2_dynamics_inference_fast_test_per_time[
                                i_rs,
                                i_continuous_states,
                                i_discrete_states,
                                i_iters,
                                trial_id,
                                :trial_length] = r2_dynamics_inference_fast_test_per_time_[cumulative_trial_length_fast:cumulative_trial_length_fast + trial_length]
                            
                            cumulative_trial_length_fast += trial_length

                        for trial_id in trial_indices_slow_test:
                            trial_length = trial_lengths_slow[trial_id] - 1

                            r2_dynamics_inference_slow_test_per_time[
                                i_rs,
                                i_continuous_states,
                                i_discrete_states,
                                i_iters,
                                trial_id,
                                :trial_length] = r2_dynamics_inference_slow_test_per_time_[cumulative_trial_length_slow:cumulative_trial_length_slow + trial_length]
                            
                            cumulative_trial_length_slow += trial_length

                        ## Compute emissions inference error
                        rmse_emissions_inference_fast_train = utils_decoding.rmse(X_emissions_inference_fast_train_concat, X_fast_train_concat)
                        rmse_emissions_inference_fast_test  = utils_decoding.rmse(X_emissions_inference_fast_test_concat,  X_fast_test_concat)
                        rmse_emissions_inference_slow_train = utils_decoding.rmse(X_emissions_inference_slow_train_concat, X_slow_train_concat)
                        rmse_emissions_inference_slow_test  = utils_decoding.rmse(X_emissions_inference_slow_test_concat,  X_slow_test_concat)
                        
                        r2_emissions_inference_fast_train = utils_decoding.r2(X_emissions_inference_fast_train_concat, X_fast_train_concat)
                        r2_emissions_inference_fast_test  = utils_decoding.r2(X_emissions_inference_fast_test_concat,  X_fast_test_concat)
                        r2_emissions_inference_slow_train = utils_decoding.r2(X_emissions_inference_slow_train_concat, X_slow_train_concat)
                        r2_emissions_inference_slow_test  = utils_decoding.r2(X_emissions_inference_slow_test_concat,  X_slow_test_concat)

                        ## Save per-time forecast error
                        r2_emissions_inference_fast_test_per_time_ = utils_decoding.r2(X_emissions_inference_fast_test_concat, X_fast_test_concat, axis=1)
                        r2_emissions_inference_slow_test_per_time_ = utils_decoding.r2(X_emissions_inference_slow_test_concat, X_slow_test_concat, axis=1)
                        
                        cumulative_trial_length_fast = 0
                        cumulative_trial_length_slow = 0

                        for trial_id in trial_indices_fast_test:
                            trial_length = trial_lengths_fast[trial_id]

                            r2_emissions_inference_fast_test_per_time[
                                i_rs,
                                i_continuous_states,
                                i_discrete_states,
                                i_iters,
                                trial_id,
                                :trial_length] = r2_emissions_inference_fast_test_per_time_[cumulative_trial_length_fast:cumulative_trial_length_fast + trial_length]
                            
                            cumulative_trial_length_fast += trial_length

                        for trial_id in trial_indices_slow_test:
                            trial_length = trial_lengths_slow[trial_id]

                            r2_emissions_inference_slow_test_per_time[
                                i_rs,
                                i_continuous_states,
                                i_discrete_states,
                                i_iters,
                                trial_id,
                                :trial_length] = r2_emissions_inference_slow_test_per_time_[cumulative_trial_length_slow:cumulative_trial_length_slow + trial_length]
                            
                            cumulative_trial_length_slow += trial_length


                        #### Save results ####                    
                        rmse_forecast_fast[0, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = rmse_forecast_fast_train
                        rmse_forecast_fast[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = rmse_forecast_fast_test
                        rmse_forecast_slow[0, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = rmse_forecast_slow_train
                        rmse_forecast_slow[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = rmse_forecast_slow_test
                        
                        r2_forecast_fast[0, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = r2_forecast_fast_train
                        r2_forecast_fast[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = r2_forecast_fast_test
                        r2_forecast_slow[0, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = r2_forecast_slow_train
                        r2_forecast_slow[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = r2_forecast_slow_test
                        
                        rmse_dynamics_inference_fast[0, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = rmse_dynamics_inference_fast_train
                        rmse_dynamics_inference_fast[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = rmse_dynamics_inference_fast_test
                        rmse_dynamics_inference_slow[0, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = rmse_dynamics_inference_slow_train
                        rmse_dynamics_inference_slow[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = rmse_dynamics_inference_slow_test
                        
                        r2_dynamics_inference_fast[0, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = r2_dynamics_inference_fast_train
                        r2_dynamics_inference_fast[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = r2_dynamics_inference_fast_test
                        r2_dynamics_inference_slow[0, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = r2_dynamics_inference_slow_train
                        r2_dynamics_inference_slow[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = r2_dynamics_inference_slow_test
                        
                        rmse_emissions_inference_fast[0, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = rmse_emissions_inference_fast_train
                        rmse_emissions_inference_fast[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = rmse_emissions_inference_fast_test
                        rmse_emissions_inference_slow[0, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = rmse_emissions_inference_slow_train
                        rmse_emissions_inference_slow[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = rmse_emissions_inference_slow_test
                        
                        r2_emissions_inference_fast[0, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = r2_emissions_inference_fast_train
                        r2_emissions_inference_fast[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = r2_emissions_inference_fast_test
                        r2_emissions_inference_slow[0, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = r2_emissions_inference_slow_train
                        r2_emissions_inference_slow[1, i_rs, i_fold, i_continuous_states, i_discrete_states, i_iters] = r2_emissions_inference_slow_test

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


    ## Save results
    np.savez(
        res_save_path,
        rmse_forecast_fast=rmse_forecast_fast,
        rmse_forecast_slow=rmse_forecast_slow,
        r2_forecast_fast=r2_forecast_fast,
        r2_forecast_slow=r2_forecast_slow,
        r2_forecast_fast_test_per_time=r2_forecast_fast_test_per_time,
        r2_forecast_slow_test_per_time=r2_forecast_slow_test_per_time,
        rmse_dynamics_inference_fast=rmse_dynamics_inference_fast,
        rmse_dynamics_inference_slow=rmse_dynamics_inference_slow,
        r2_dynamics_inference_fast=r2_dynamics_inference_fast,
        r2_dynamics_inference_slow=r2_dynamics_inference_slow,
        r2_dynamics_inference_fast_test_per_time=r2_dynamics_inference_fast_test_per_time,
        r2_dynamics_inference_slow_test_per_time=r2_dynamics_inference_slow_test_per_time,
        rmse_emissions_inference_fast=rmse_emissions_inference_fast,
        rmse_emissions_inference_slow=rmse_emissions_inference_slow,
        r2_emissions_inference_fast=r2_emissions_inference_fast,
        r2_emissions_inference_slow=r2_emissions_inference_slow,
        r2_emissions_inference_fast_test_per_time=r2_emissions_inference_fast_test_per_time,
        r2_emissions_inference_slow_test_per_time=r2_emissions_inference_slow_test_per_time,
        forecast_fast_test=forecast_fast_test,
        forecast_slow_test=forecast_slow_test,
        dynamics_inference_fast_test=dynamics_inference_fast_test,
        dynamics_inference_slow_test=dynamics_inference_slow_test,
        emissions_inference_fast_test=emissions_inference_fast_test,
        emissions_inference_slow_test=emissions_inference_slow_test,
        trial_lengths_fast=trial_lengths_fast,
        trial_lengths_slow=trial_lengths_slow,
        ns_states=ns_states,
        ns_iters=ns_iters,
        random_states=random_states,
        n_folds=n_folds)



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
        print('Running Inference ...')
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
    