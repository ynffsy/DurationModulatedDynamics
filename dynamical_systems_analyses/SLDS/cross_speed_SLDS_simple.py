import os
import time
import ipdb
import pickle
import datetime

import numpy as np
import pandas as pd

import config as config
import utils.utils_processing as utils_processing
from SLDS import SLDS

        

## Read parameters from config
overwrite_results  = config.overwrite_results
data_dir           = config.data_dir
results_dir        = config.results_dir
vis_dir            = config.vis_dir
session_data_names = config.session_data_names
trial_filters      = config.trial_filters

unit_filters       = config.unit_filters
input_unit_filters = config.input_unit_filters
window_config      = config.window_config
time_offset        = config.time_offset
random_states      = config.random_states
ns_states          = config.ns_states
n_discrete_states  = config.n_discrete_states
ns_iters           = config.ns_iters
batch_size         = config.batch_size 

model_type    = config.model_type
method_type   = config.method_type
init_type     = config.init_type
data_format   = config.data_format



def main(
    session_data_name,
    unit_filter,
    input_unit_filter,
    trial_filter):

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
        time_offset,
        'cross_speed',
        model_type,
        n_discrete_states,
        method_type,
        init_type,
        data_format,
        check_existence=False)

    ## Save runtimes
    runtimes = np.zeros((len(random_states), len(ns_states), len(ns_iters)))

    for i_rs, random_state in enumerate(random_states):

        np.random.seed(random_state)

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('random state: ', random_state)

        ## Sweep through various numbers of states and iterations as well as random states
        for i_states, n_continuous_states in enumerate(ns_states):
            for i_iters, n_iters in enumerate(ns_iters):

                time_start = time.time()

                ## Format save path
                model_save_name = '_'.join(map(str, [x for x in [
                    'r' + str(random_state),
                    's' + str(n_continuous_states),
                    'i' + str(n_iters)]]))
                
                model_save_prefix = os.path.join(model_results_dir, model_save_name)
                res_save_path     = model_save_prefix + '.pkl'

                if (not overwrite_results) and os.path.isfile(res_save_path):
                    print('s' + str(n_continuous_states) + 'i' + str(n_iters) + ' model already exists. Skipping...')
                    continue

                X_input_train = None

                ## Use SLDS to reduce the dimensionality of firing rate data                    
                neural_SLDS = SLDS(
                    firing_rates_self_simple,
                    X_input_train,
                    model_type,
                    init_type,
                    n_neurons,
                    n_input_neurons,
                    n_continuous_states,
                    n_discrete_states,
                    n_iters,
                    batch_size,
                    data_format,
                    random_state)
                
                neural_SLDS.fit()
                neural_SLDS.transform(test_emissions=firing_rates_ctpt_simple)

                ## Save fitted params and NLLs over iterations
                # with open(params_save_path, 'wb') as f:
                #     pickle.dump(neural_SSM.params, f)
                
                # with open(nll_save_path, 'wb') as f:
                #     pickle.dump([neural_SSM.neg_marginal_lls, neural_SSM.train_emissions.size], f)

                # ## Use fitted SSM to transform train and test data and save results
                # X_train_latent, X_train_latent_cov = neural_SSM.transform(trial_lengths=trial_lengths_train)
                # X_test_latent,  X_test_latent_cov  = neural_SSM.transform(test_emissions=X_test, trial_lengths=trial_lengths_test)

                with open(res_save_path , 'wb') as f:
                    pickle.dump({
                        'train_elbos'             : neural_SLDS.train_elbos,
                        'train_continuous_states' : neural_SLDS.train_continuous_states,
                        'train_discrete_states'   : neural_SLDS.train_discrete_states,
                        'test_elbos'              : neural_SLDS.test_elbos,
                        'test_continuous_states'  : neural_SLDS.test_continuous_states,
                        'test_discrete_states'    : neural_SLDS.test_discrete_states,
                        'model'                   : neural_SLDS.model,
                    }, f)

                time_end = time.time()
                runtime = time_end - time_start
                runtimes[i_rs, i_states, i_iters] = runtime

                print('# continuous states: ', n_continuous_states, ' # iters: ', n_iters, ' run time: ', runtime)

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


    ## Save runtimes
    ## (Not saving the results when all models are skipped)
    if not np.all(runtimes == 0):

        metadata_save_name = 'metadata_' + datetime.datetime.now().strftime('%m%d%Y-%H%M%S')    
        metadata_save_path = os.path.join(model_results_dir, metadata_save_name + '.pkl')

        with open(metadata_save_path, 'wb') as f:
            pickle.dump({
                'n_neurons'     : n_neurons,
                'n_trials'      : n_trials_self,
                'random_states' : random_states,
                'ns_states'     : ns_states,
                'ns_iters'      : ns_iters,
                'batch_size'    : batch_size,
                'data_format'   : data_format,
                'runtimes'      : runtimes,
            }, f)



if __name__ == '__main__':

    for session_data_name in session_data_names:
        for unit_filter in unit_filters:
            for input_unit_filter in input_unit_filters:
                for trial_filter in trial_filters:

                    print('=============================================================')
                    print('Running Cross Speed SSM')
                    print('\tsession_data_name: ', session_data_name)
                    print('\ttrial_filter: ',      trial_filter)
                    print('\tunit_filter: ',       unit_filter)
                    print('\tmodel_type: ',        model_type) 
                    print('\tmethod_type: ',       method_type)
                    print('\tdata_format: ',       data_format)
                    print('\tbatch_size: ',        batch_size)
                    print('\trandom_states: ',     random_states)
                    print('\tns_states: ',         ns_states)
                    print('\tns_iters: ',          ns_iters)
                    print('=============================================================')

                    main(session_data_name, unit_filter, input_unit_filter, trial_filter)
