import os
import ipdb
import pickle
import itertools

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

import dynamical_systems_analyses.SLDS.config as config
import dynamical_systems_analyses.utils.utils_processing as utils_processing
from vis_config import session_target_radii



## Read parameters from config
data_dir           = config.data_dir
results_dir        = config.results_dir
vis_dir            = config.vis_dir
session_data_names = config.session_data_names
train_test_options = config.train_test_options

unit_filters       = config.unit_filters
input_unit_filters = config.input_unit_filters
window_config      = config.window_config
time_offset        = config.time_offset
trial_filters      = config.trial_filters
random_states      = config.random_states
n_folds            = config.n_folds
ns_states          = config.ns_states
ns_discrete_states = config.ns_discrete_states
ns_iters           = config.ns_iters
window_sizes       = config.window_sizes

model_types        = config.model_types
dynamics_classes   = config.dynamics_classes
emission_classes   = config.emission_classes
init_types         = config.init_types
data_formats       = config.data_formats
standardize        = config.standardize
label_formats      = config.label_formats



def compile_decoding_SSM_results(
    session_data_name,
    unit_filter,
    input_unit_filter,
    window_config,
    trial_filters,
    train_test,
    label_format):

    ## Load directories
    data_loader = utils_processing.DataLoaderDuo(
        data_dir,
        results_dir,
        session_data_name,
        unit_filter,
        input_unit_filter,
        window_config,
        trial_filters)

    slow_model_results_dir, fast_model_results_dir = data_loader.get_model_result_dirs(
        time_offset,
        train_test,
        model_type,
        n_discrete_states,
        method_type,
        init_type,
        data_format)    

    ## Initialize results
    if label_format == 'cartesian':
        n_labels       = 4
        n_labels_final = 6  ## will include average position and velocity
    elif label_format == 'polar':
        n_labels       = 4
        n_labels_final = 4
    elif label_format == 'uvd2tc' or label_format == 'uvd2tp':
        n_labels       = 2 
        n_labels_final = 2
    else:
        raise ValueError('Invalid label format: ', label_format)
    
    res_SSM_train_slow = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_iters), n_labels_final))
    res_SSM_train_fast = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_iters), n_labels_final))

    predictions_SSM_slow_latent_train = []
    predictions_SSM_slow_latent_test  = []
    predictions_SSM_fast_latent_train = []
    predictions_SSM_fast_latent_test  = []

    ## Read decoding results
    for i_rs, random_state in enumerate(random_states):
        for i_fold in range(n_folds):
            for i_states, n_states in enumerate(ns_states):
                for i_iters, n_iters in enumerate(ns_iters):

                    print('Compiling decoding results for random state: ', random_state, ' fold:', i_fold, ' n_states:', n_states, ' n_iters:', n_iters)

                    SSM_save_name = '_'.join(map(str, [x for x in [
                        'r' + str(random_state),
                        'f' + str(i_fold),
                        's' + str(n_states),
                        'i' + str(n_iters)]]))
                    
                    decoding_SSM_slow_save_path_prefix = os.path.join(slow_model_results_dir, SSM_save_name + '_decoding')
                    decoding_SSM_fast_save_path_prefix = os.path.join(fast_model_results_dir, SSM_save_name + '_decoding')

                    if standardize:
                        decoding_SSM_slow_save_path_prefix += '_standardize'
                        decoding_SSM_fast_save_path_prefix += '_standardize'

                    decoding_SSM_slow_save_path = decoding_SSM_slow_save_path_prefix + '.npz'
                    decoding_SSM_fast_save_path = decoding_SSM_fast_save_path_prefix + '.npz'

                    decoding_SSM_slow = np.load(decoding_SSM_slow_save_path)
                    decoding_SSM_fast = np.load(decoding_SSM_fast_save_path)

                    res_SSM_train_slow[0, i_rs, i_fold, i_states, i_iters, :n_labels] = decoding_SSM_slow['rmse_train']
                    res_SSM_train_slow[1, i_rs, i_fold, i_states, i_iters, :n_labels] = decoding_SSM_slow['rmse_test']
                    res_SSM_train_fast[0, i_rs, i_fold, i_states, i_iters, :n_labels] = decoding_SSM_fast['rmse_train']
                    res_SSM_train_fast[1, i_rs, i_fold, i_states, i_iters, :n_labels] = decoding_SSM_fast['rmse_test']

                    predictions_SSM_slow_latent_train.append(decoding_SSM_slow['train_pred'])
                    predictions_SSM_slow_latent_test.append(decoding_SSM_slow['test_pred'])
                    predictions_SSM_fast_latent_train.append(decoding_SSM_fast['train_pred'])
                    predictions_SSM_fast_latent_test.append(decoding_SSM_fast['test_pred'])


    if label_format == 'cartesian':

        ## Compute average position and velocity
        res_SSM_train_slow[:, :, :, :, :, 4] = np.mean(res_SSM_train_slow[:, :, :, :, :, 0:2], axis=-1)
        res_SSM_train_slow[:, :, :, :, :, 5] = np.mean(res_SSM_train_slow[:, :, :, :, :, 2:4], axis=-1)
        res_SSM_train_fast[:, :, :, :, :, 4] = np.mean(res_SSM_train_fast[:, :, :, :, :, 0:2], axis=-1)
        res_SSM_train_fast[:, :, :, :, :, 5] = np.mean(res_SSM_train_fast[:, :, :, :, :, 2:4], axis=-1)

    ## Concatenate predictions
    predictions_SSM_slow_latent_train = np.concatenate(predictions_SSM_slow_latent_train, axis=0)
    predictions_SSM_slow_latent_test  = np.concatenate(predictions_SSM_slow_latent_test, axis=0)
    predictions_SSM_fast_latent_train = np.concatenate(predictions_SSM_fast_latent_train, axis=0)
    predictions_SSM_fast_latent_test  = np.concatenate(predictions_SSM_fast_latent_test, axis=0)

    ## Save results
    res_save_name = '_'.join(map(str, [x for x in [
        'decoding',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        trial_filters,
        train_test,
        model_type,
        n_discrete_states,
        method_type,
        init_type,
        data_format,
        label_format] if x is not None]))

    if standardize:
        res_save_name += '_standardize'

    session_results_dir = os.path.join(results_dir, session_data_name)

    np.savez(
        os.path.join(session_results_dir, res_save_name + '.npz'),
        res_SSM_train_slow=res_SSM_train_slow,
        res_SSM_train_fast=res_SSM_train_fast,
        predictions_SSM_slow_latent_train=predictions_SSM_slow_latent_train,
        predictions_SSM_slow_latent_test=predictions_SSM_slow_latent_test,
        predictions_SSM_fast_latent_train=predictions_SSM_fast_latent_train,
        predictions_SSM_fast_latent_test=predictions_SSM_fast_latent_test,
        random_states=random_states,
        n_folds=n_folds,
        ns_states=ns_states,
        ns_iters=ns_iters,
    )


def compile_decoding_baseline_results(
    session_data_name,
    unit_filter,
    input_unit_filter,
    window_config,
    trial_filters,
    train_test,
    label_format):

    ## Load directories
    data_loader = utils_processing.DataLoaderDuo(
        data_dir,
        results_dir,
        session_data_name,
        unit_filter,
        input_unit_filter,
        window_config,
        trial_filters)
    
    trial_filters = data_loader.trial_filters

    slow_model_results_dir, fast_model_results_dir = data_loader.get_model_result_dirs(
        time_offset=time_offset,
        train_test=train_test,
        model_type='baselines',
        n_discrete_states=None,
        method_type=None,
        init_type=None,
        data_format=data_format,
        check_existence=False)    

    ## Initialize results
    if label_format == 'cartesian':
        n_labels       = 4
        n_labels_final = 6  ## will include average position and velocity
    elif label_format == 'polar':
        n_labels       = 4
        n_labels_final = 4
    elif label_format == 'uvd2tc' or label_format == 'uvd2tp':
        n_labels       = 2 
        n_labels_final = 2
    else:
        raise ValueError('Invalid label format: ', label_format)
    
    res_PCA_train_slow      = np.zeros((2, len(random_states), n_folds, len(ns_states), n_labels_final))
    res_PCA_train_fast      = np.zeros((2, len(random_states), n_folds, len(ns_states), n_labels_final))
    res_FA_train_slow       = np.zeros((2, len(random_states), n_folds, len(ns_states), n_labels_final))
    res_FA_train_fast       = np.zeros((2, len(random_states), n_folds, len(ns_states), n_labels_final))
    res_PLSR_train_slow     = np.zeros((2, len(random_states), n_folds, len(ns_states), n_labels_final))
    res_PLSR_train_fast     = np.zeros((2, len(random_states), n_folds, len(ns_states), n_labels_final))
    res_baseline_train_slow = np.zeros((2, len(random_states), n_folds, n_labels_final))
    res_baseline_train_fast = np.zeros((2, len(random_states), n_folds, n_labels_final))

    predictions_PCA_slow_latent_train = []
    predictions_PCA_slow_latent_test  = []
    predictions_PCA_fast_latent_train = []
    predictions_PCA_fast_latent_test  = []

    predictions_FA_slow_latent_train = []
    predictions_FA_slow_latent_test  = []
    predictions_FA_fast_latent_train = []
    predictions_FA_fast_latent_test  = []

    predictions_PLSR_slow_latent_train = []
    predictions_PLSR_slow_latent_test  = []
    predictions_PLSR_fast_latent_train = []
    predictions_PLSR_fast_latent_test  = []

    predictions_baseline_slow_latent_train = []
    predictions_baseline_slow_latent_test  = []
    predictions_baseline_fast_latent_train = []
    predictions_baseline_fast_latent_test  = []

    ## Read decoding results
    for i_rs, random_state in enumerate(random_states):
        for i_fold in range(n_folds):
            for i_states, n_states in enumerate(ns_states):

                print('Compiling decoding results for random state: ', random_state, ' fold:', i_fold, ' n_states:', n_states)

                model_save_name = '_'.join(map(str, [x for x in [
                    'r' + str(random_state),
                    'f' + str(i_fold),
                    's' + str(n_states)]]))
                
                decoding_PCA_slow_save_path_prefix      = os.path.join(slow_model_results_dir, model_save_name + '_PCA')
                decoding_PCA_fast_save_path_prefix      = os.path.join(fast_model_results_dir, model_save_name + '_PCA')
                decoding_FA_slow_save_path_prefix       = os.path.join(slow_model_results_dir, model_save_name + '_FA')
                decoding_FA_fast_save_path_prefix       = os.path.join(fast_model_results_dir, model_save_name + '_FA')
                decoding_PLSR_slow_save_path_prefix     = os.path.join(slow_model_results_dir, model_save_name + '_PLSR')
                decoding_PLSR_fast_save_path_prefix     = os.path.join(fast_model_results_dir, model_save_name + '_PLSR')
                decoding_baseline_slow_save_path_prefix = os.path.join(slow_model_results_dir, model_save_name + '_baseline')
                decoding_baseline_fast_save_path_prefix = os.path.join(fast_model_results_dir, model_save_name + '_baseline')

                if standardize:
                    decoding_PCA_slow_save_path_prefix      += '_standardize'
                    decoding_PCA_fast_save_path_prefix      += '_standardize'
                    decoding_FA_slow_save_path_prefix       += '_standardize'
                    decoding_FA_fast_save_path_prefix       += '_standardize'
                    decoding_PLSR_slow_save_path_prefix     += '_standardize'
                    decoding_PLSR_fast_save_path_prefix     += '_standardize'
                    decoding_baseline_slow_save_path_prefix += '_standardize'
                    decoding_baseline_fast_save_path_prefix += '_standardize'

                decoding_PCA_slow_save_path      = decoding_PCA_slow_save_path_prefix      + '.npz'
                decoding_PCA_fast_save_path      = decoding_PCA_fast_save_path_prefix      + '.npz'
                decoding_FA_slow_save_path       = decoding_FA_slow_save_path_prefix       + '.npz'
                decoding_FA_fast_save_path       = decoding_FA_fast_save_path_prefix       + '.npz'
                decoding_PLSR_slow_save_path     = decoding_PLSR_slow_save_path_prefix     + '.npz'
                decoding_PLSR_fast_save_path     = decoding_PLSR_fast_save_path_prefix     + '.npz'
                decoding_baseline_slow_save_path = decoding_baseline_slow_save_path_prefix + '.npz'
                decoding_baseline_fast_save_path = decoding_baseline_fast_save_path_prefix + '.npz'

                decoding_PCA_slow      = np.load(decoding_PCA_slow_save_path)
                decoding_PCA_fast      = np.load(decoding_PCA_fast_save_path)
                decoding_FA_slow       = np.load(decoding_FA_slow_save_path)
                decoding_FA_fast       = np.load(decoding_FA_fast_save_path)
                decoding_PLSR_slow     = np.load(decoding_PLSR_slow_save_path)
                decoding_PLSR_fast     = np.load(decoding_PLSR_fast_save_path)
                # decoding_baseline_slow = np.load(decoding_baseline_slow_save_path)
                # decoding_baseline_fast = np.load(decoding_baseline_fast_save_path)

                res_PCA_train_slow[0, i_rs, i_fold, i_states, :n_labels] = decoding_PCA_slow['rmse_train']
                res_PCA_train_slow[1, i_rs, i_fold, i_states, :n_labels] = decoding_PCA_slow['rmse_test']
                res_PCA_train_fast[0, i_rs, i_fold, i_states, :n_labels] = decoding_PCA_fast['rmse_train']
                res_PCA_train_fast[1, i_rs, i_fold, i_states, :n_labels] = decoding_PCA_fast['rmse_test']

                predictions_PCA_slow_latent_train.append(decoding_PCA_slow['train_pred'])
                predictions_PCA_slow_latent_test.append(decoding_PCA_slow['test_pred'])
                predictions_PCA_fast_latent_train.append(decoding_PCA_fast['train_pred'])
                predictions_PCA_fast_latent_test.append(decoding_PCA_fast['test_pred'])

                res_FA_train_slow[0, i_rs, i_fold, i_states, :n_labels] = decoding_FA_slow['rmse_train']
                res_FA_train_slow[1, i_rs, i_fold, i_states, :n_labels] = decoding_FA_slow['rmse_test']
                res_FA_train_fast[0, i_rs, i_fold, i_states, :n_labels] = decoding_FA_fast['rmse_train']
                res_FA_train_fast[1, i_rs, i_fold, i_states, :n_labels] = decoding_FA_fast['rmse_test']

                predictions_FA_slow_latent_train.append(decoding_FA_slow['train_pred'])
                predictions_FA_slow_latent_test.append(decoding_FA_slow['test_pred'])
                predictions_FA_fast_latent_train.append(decoding_FA_fast['train_pred'])
                predictions_FA_fast_latent_test.append(decoding_FA_fast['test_pred'])

                res_PLSR_train_slow[0, i_rs, i_fold, i_states, :n_labels] = decoding_PLSR_slow['rmse_train']
                res_PLSR_train_slow[1, i_rs, i_fold, i_states, :n_labels] = decoding_PLSR_slow['rmse_test']
                res_PLSR_train_fast[0, i_rs, i_fold, i_states, :n_labels] = decoding_PLSR_fast['rmse_train']
                res_PLSR_train_fast[1, i_rs, i_fold, i_states, :n_labels] = decoding_PLSR_fast['rmse_test']

                predictions_PLSR_slow_latent_train.append(decoding_PLSR_slow['train_pred'])
                predictions_PLSR_slow_latent_test.append(decoding_PLSR_slow['test_pred'])
                predictions_PLSR_fast_latent_train.append(decoding_PLSR_fast['train_pred'])
                predictions_PLSR_fast_latent_test.append(decoding_PLSR_fast['test_pred'])

                # res_baseline_train_slow[0, i_rs, i_fold, :n_labels] = decoding_baseline_slow['rmse_train']
                # res_baseline_train_slow[1, i_rs, i_fold, :n_labels] = decoding_baseline_slow['rmse_test']
                # res_baseline_train_fast[0, i_rs, i_fold, :n_labels] = decoding_baseline_fast['rmse_train']
                # res_baseline_train_fast[1, i_rs, i_fold, :n_labels] = decoding_baseline_fast['rmse_test']

                # predictions_baseline_slow_latent_train.append(decoding_baseline_slow['train_pred'])
                # predictions_baseline_slow_latent_test.append(decoding_baseline_slow['test_pred'])
                # predictions_baseline_fast_latent_train.append(decoding_baseline_fast['train_pred'])
                # predictions_baseline_fast_latent_test.append(decoding_baseline_fast['test_pred'])

    if label_format == 'cartesian':

        ## Compute average position and velocity
        res_PCA_train_slow[:, :, :, :, 4] = np.mean(res_PCA_train_slow[:, :, :, :, 0:2], axis=-1)
        res_PCA_train_slow[:, :, :, :, 5] = np.mean(res_PCA_train_slow[:, :, :, :, 2:4], axis=-1)
        res_PCA_train_fast[:, :, :, :, 4] = np.mean(res_PCA_train_fast[:, :, :, :, 0:2], axis=-1)
        res_PCA_train_fast[:, :, :, :, 5] = np.mean(res_PCA_train_fast[:, :, :, :, 2:4], axis=-1)

        res_FA_train_slow[:, :, :, :, 4] = np.mean(res_FA_train_slow[:, :, :, :, 0:2], axis=-1)
        res_FA_train_slow[:, :, :, :, 5] = np.mean(res_FA_train_slow[:, :, :, :, 2:4], axis=-1)
        res_FA_train_fast[:, :, :, :, 4] = np.mean(res_FA_train_fast[:, :, :, :, 0:2], axis=-1)
        res_FA_train_fast[:, :, :, :, 5] = np.mean(res_FA_train_fast[:, :, :, :, 2:4], axis=-1)

        res_PLSR_train_slow[:, :, :, :, 4] = np.mean(res_PLSR_train_slow[:, :, :, :, 0:2], axis=-1)
        res_PLSR_train_slow[:, :, :, :, 5] = np.mean(res_PLSR_train_slow[:, :, :, :, 2:4], axis=-1)
        res_PLSR_train_fast[:, :, :, :, 4] = np.mean(res_PLSR_train_fast[:, :, :, :, 0:2], axis=-1)
        res_PLSR_train_fast[:, :, :, :, 5] = np.mean(res_PLSR_train_fast[:, :, :, :, 2:4], axis=-1)

        # res_baseline_train_slow[:, :, :, 4] = np.mean(res_baseline_train_slow[:, :, :, 0:2], axis=-1)
        # res_baseline_train_slow[:, :, :, 5] = np.mean(res_baseline_train_slow[:, :, :, 2:4], axis=-1)
        # res_baseline_train_fast[:, :, :, 4] = np.mean(res_baseline_train_fast[:, :, :, 0:2], axis=-1)
        # res_baseline_train_fast[:, :, :, 5] = np.mean(res_baseline_train_fast[:, :, :, 2:4], axis=-1)


    ## Concatenate predictions
    predictions_PCA_slow_latent_train = np.concatenate(predictions_PCA_slow_latent_train, axis=0)
    predictions_PCA_slow_latent_test  = np.concatenate(predictions_PCA_slow_latent_test, axis=0)
    predictions_PCA_fast_latent_train = np.concatenate(predictions_PCA_fast_latent_train, axis=0)
    predictions_PCA_fast_latent_test  = np.concatenate(predictions_PCA_fast_latent_test, axis=0)

    predictions_FA_slow_latent_train = np.concatenate(predictions_FA_slow_latent_train, axis=0)
    predictions_FA_slow_latent_test  = np.concatenate(predictions_FA_slow_latent_test, axis=0)
    predictions_FA_fast_latent_train = np.concatenate(predictions_FA_fast_latent_train, axis=0)
    predictions_FA_fast_latent_test  = np.concatenate(predictions_FA_fast_latent_test, axis=0)

    predictions_PLSR_slow_latent_train = np.concatenate(predictions_PLSR_slow_latent_train, axis=0)
    predictions_PLSR_slow_latent_test  = np.concatenate(predictions_PLSR_slow_latent_test, axis=0)
    predictions_PLSR_fast_latent_train = np.concatenate(predictions_PLSR_fast_latent_train, axis=0)
    predictions_PLSR_fast_latent_test  = np.concatenate(predictions_PLSR_fast_latent_test, axis=0)

    # predictions_baseline_slow_latent_train = np.concatenate(predictions_baseline_slow_latent_train, axis=0)
    # predictions_baseline_slow_latent_test  = np.concatenate(predictions_baseline_slow_latent_test, axis=0)
    # predictions_baseline_fast_latent_train = np.concatenate(predictions_baseline_fast_latent_train, axis=0)
    # predictions_baseline_fast_latent_test  = np.concatenate(predictions_baseline_fast_latent_test, axis=0)

    ## Save results
    res_save_name = '_'.join(map(str, [x for x in [
        'decoding_baseline',
        unit_filter,
        trial_filters,
        train_test,
        data_format,
        label_format] if x is not None]))

    if standardize:
        res_save_name += '_standardize'

    session_results_dir = os.path.join(results_dir, session_data_name)
    
    np.savez(
        os.path.join(session_results_dir, res_save_name + '.npz'),
        res_PCA_train_slow=res_PCA_train_slow,
        res_PCA_train_fast=res_PCA_train_fast,
        res_FA_train_slow=res_FA_train_slow,
        res_FA_train_fast=res_FA_train_fast,
        res_PLSR_train_slow=res_PLSR_train_slow,
        res_PLSR_train_fast=res_PLSR_train_fast,
        res_baseline_train_slow=res_baseline_train_slow,
        res_baseline_train_fast=res_baseline_train_fast,
        predictions_PCA_slow_latent_train=predictions_PCA_slow_latent_train,
        predictions_PCA_slow_latent_test=predictions_PCA_slow_latent_test,
        predictions_PCA_fast_latent_train=predictions_PCA_fast_latent_train,
        predictions_PCA_fast_latent_test=predictions_PCA_fast_latent_test,
        predictions_FA_slow_latent_train=predictions_FA_slow_latent_train,
        predictions_FA_slow_latent_test=predictions_FA_slow_latent_test,
        predictions_FA_fast_latent_train=predictions_FA_fast_latent_train,
        predictions_FA_fast_latent_test=predictions_FA_fast_latent_test,
        predictions_PLSR_slow_latent_train=predictions_PLSR_slow_latent_train,
        predictions_PLSR_slow_latent_test=predictions_PLSR_slow_latent_test,
        predictions_PLSR_fast_latent_train=predictions_PLSR_fast_latent_train,
        predictions_PLSR_fast_latent_test=predictions_PLSR_fast_latent_test,
        predictions_baseline_slow_latent_train=predictions_baseline_slow_latent_train,
        predictions_baseline_slow_latent_test=predictions_baseline_slow_latent_test,
        predictions_baseline_fast_latent_train=predictions_baseline_fast_latent_train,
        predictions_baseline_fast_latent_test=predictions_baseline_fast_latent_test,
        random_states=random_states,
        n_folds=n_folds,
        ns_states=ns_states,
    )


def compile_dsup_ratio_results(
    session_data_name,
    unit_filter,
    train_test,
    trial_filters=None):

    ## Load directories
    data_loader = utils_processing.DataLoaderDuo(
        data_dir,
        results_dir,
        session_data_name,
        unit_filter,
        trial_filters)
    
    trial_filters = data_loader.trial_filters

    slow_model_results_dir, fast_model_results_dir = data_loader.get_model_result_dirs(
        train_test,
        model_type,
        method_type,
        init_type,
        data_format)    

    ## Initialize results
    ## The 1st dimension has 2 elements, train and test
    dsupr_SSM_slow_all = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_iters)))
    dsupr_SSM_fast_all = np.zeros((2, len(random_states), n_folds, len(ns_states), len(ns_iters)))

    ## Read DSUPR results
    for i_rs, random_state in enumerate(random_states):
        for i_fold in range(n_folds):
            for i_states, n_states in enumerate(ns_states):
                for i_iters, n_iters in enumerate(ns_iters):

                    print('Compiling DSUPR results for random state: ', random_state, ' fold:', i_fold, ' n_states:', n_states, ' n_iters:', n_iters)

                    SSM_save_name = '_'.join(map(str, [x for x in [
                        'r' + str(random_state),
                        'f' + str(i_fold),
                        's' + str(n_states),
                        'i' + str(n_iters)]]))
                    
                    dsupr_SSM_slow_save_path_prefix = os.path.join(slow_model_results_dir, SSM_save_name + '_dsupr')
                    dsupr_SSM_fast_save_path_prefix = os.path.join(fast_model_results_dir, SSM_save_name + '_dsupr')

                    dsupr_SSM_slow_save_path = dsupr_SSM_slow_save_path_prefix + '.npz'
                    dsupr_SSM_fast_save_path = dsupr_SSM_fast_save_path_prefix + '.npz'

                    dsupr_SSM_slow = np.load(dsupr_SSM_slow_save_path)
                    dsupr_SSM_fast = np.load(dsupr_SSM_fast_save_path)

                    dsupr_SSM_slow_all[0, i_rs, i_fold, i_states, i_iters] = dsupr_SSM_slow['dsupr_train']
                    dsupr_SSM_slow_all[1, i_rs, i_fold, i_states, i_iters] = dsupr_SSM_slow['dsupr_test']
                    dsupr_SSM_fast_all[0, i_rs, i_fold, i_states, i_iters] = dsupr_SSM_fast['dsupr_train']
                    dsupr_SSM_fast_all[1, i_rs, i_fold, i_states, i_iters] = dsupr_SSM_fast['dsupr_test']

    ## Save results
    res_save_name = '_'.join(map(str, [x for x in [
        'dsupr',
        unit_filter,
        trial_filters,
        train_test,
        model_type,
        method_type,
        init_type,
        data_format] if x is not None]))

    session_results_dir = os.path.join(results_dir, session_data_name)

    np.savez(
        os.path.join(session_results_dir, res_save_name + '.npz'),
        dsupr_SSM_slow=dsupr_SSM_slow_all,
        dsupr_SSM_fast=dsupr_SSM_fast_all,
        ns_states=ns_states,
        ns_iters=ns_iters,
        random_states=random_states,
        n_folds=n_folds,
        model_type=model_type,
        method_type=method_type,
        init_type=init_type,
    )



def compile_trajectories(
    session_data_name,
    unit_filter,
    input_unit_filter,
    data_format,
    trial_filter):

    session_results_dir = os.path.join(results_dir, session_data_name)

    if not os.path.exists(session_results_dir):
        os.makedirs(session_results_dir)

    data_loader = utils_processing.DataLoader(
        data_dir,
        results_dir,
        session_data_name,
        unit_filter,
        input_unit_filter,
        window_config,
        trial_filter)

    data_loader.load_firing_rate_data()

    (firing_rates_simple, 
     input_firing_rates_simple,
     trial_ids, 
     n_trials, 
     trial_lengths, 
     times_new, _) = data_loader.reformat_firing_rate_data(data_format)

    target_ids = data_loader.get_target_ids()
    target_pos = data_loader.get_target_positions()
    data_loader.load_cursor_data()
    data_loader.remove_target_overlap(target_radius=session_target_radii[session_data_name])
    cursor_states = data_loader.align_cursor_to_firing_rates()

    cursor_positions = [cursor_states[i_trial][:, 0:2] for i_trial in range(n_trials)]

    ## Train PCA
    X_concat = np.concatenate(firing_rates_simple, axis=0)

    pca                 = PCA(n_components=2)
    X_latent_PCA_concat = pca.fit_transform(X_concat)

    X_latent_PCA = []
    trial_length_sum = 0
    for i_trial in range(n_trials):
        X_latent_PCA.append(X_latent_PCA_concat[trial_length_sum : trial_length_sum + trial_lengths[i_trial], :])
        trial_length_sum += trial_lengths[i_trial]

    ## Retrieve SSM results
    # model_dir_name = '_'.join(map(str, [x for x in [
    #     unit_filter,
    #     trial_filter,
    #     'same_speed',
    #     model_type,
    #     method_type,
    #     data_format] if x is not None]))

    # model_results_dir = os.path.join(session_results_dir, model_dir_name)
    
    # save_name_SSM = '_'.join(map(str, [x for x in [
    #     'r' + str(random_state),
    #     's' + str(n_states),
    #     'i' + str(n_iters)]]))
    
    # save_path_SSM = os.path.join(model_results_dir, save_name_SSM + '.pkl')

    # with open(save_path_SSM, 'rb') as f:
    #     X_latent_SSM_ = pickle.load(f)

    # X_latent_SSM = X_latent_SSM_['X_train_latent'].reshape((n_trials, n_times, 2))

    # assert X_latent_SSM.shape == X_latent_PCA.shape
    # assert X_latent_SSM.shape == cursor_pos_all.shape

    ## Save compiled trajectories
    save_name = '_'.join(map(str, [x for x in [
        'compiled_trajectories',
        unit_filter,
        window_config,
        trial_filter,
        data_format] if x is not None]))
    
    # np.savez(os.path.join(session_results_dir, save_name + '.npz'),
    #     trial_ids=trial_ids,
    #     target_ids=target_ids,
    #     target_pos=target_pos,
    #     cursor_pos=cursor_pos_all,
    #     X_latent_PCA=X_latent_PCA,
    #     X_latent_SSM=X_latent_SSM)
    
    np.savez(os.path.join(session_results_dir, save_name + '.npz'),
        trial_ids=trial_ids,
        target_ids=target_ids,
        target_pos=target_pos,
        cursor_pos=cursor_positions,
        X_latent_PCA=X_latent_PCA)



if __name__ == '__main__':

    for (
        session_data_name, 
        unit_filter, 
        input_unit_filter,
        data_format,
        trial_filter) in itertools.product(
            session_data_names, 
            unit_filters, 
            input_unit_filters,
            data_formats,
            trial_filters):

        compile_trajectories(
            session_data_name,
            unit_filter,
            input_unit_filter,
            data_format,
            trial_filter)



    # for session_data_name in session_data_names:
    #     for unit_filter in unit_filters:
    #         for input_unit_filter in input_unit_filters:
    #             for trial_filter in trial_filters:

    #                 compile_trajectories(
    #                     session_data_name, 
    #                     unit_filter, 
    #                     trial_filter,
    #                     n_states=2,
    #                     n_iters=100,
    #                     random_state=random_states[0])

    #                 pass

                
                # for train_test in train_test_options:
                #     for label_format in label_formats:

                #         compile_decoding_SSM_results(
                #             session_data_name,
                #             unit_filter,
                #             input_unit_filter,
                #             window_config,
                #             trial_filters,
                #             train_test,
                #             label_format)

                #         # compile_decoding_baseline_results(
                #         #     session_data_name,
                #         #     unit_filter,
                #         #     input_unit_filter,
                #         #     window_config,
                #         #     trial_filters,
                #         #     train_test,
                #         #     label_format)

                #         # compile_dsup_ratio_results(
                #         #     session_data_name,
                #         #     unit_filter,
                #         #     train_test)

                #         pass
                