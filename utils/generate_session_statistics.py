"""Quick reporter for neuron counts/trial counts per session configuration."""

import os
import ipdb
import numpy as np
import pandas as pd

import scripts.config as config
import utils.utils_processing as utils_processing



## Read parameters from config
data_dir          = config.data_dir
results_dir       = config.results_dir
session_data_dict = config.session_data_dict

# window_config = 'gt_0.0_fct_0.0_s0.02_gaussian_0.1_10'
window_config = 'gt_-0.2_fct_0.5_s0.001_gaussian_0.03_10'



def generate_session_statistics(
    session_data_name):
    """Print subject/task summary plus neuron/trial counts for a session."""

    session_data_name_list = session_data_name.split('_')
    subject_id = session_data_name_list[0][-2:]
    session_date = session_data_name_list[1][-8:]
    task = session_data_name_list[-1]

    unit_filters = session_data_dict[session_data_name]['unit_filters']
    input_unit_filters = session_data_dict[session_data_name]['input_unit_filters']
    trial_filters = session_data_dict[session_data_name]['trial_filters']

    n_neurons = []

    for unit_filter, input_unit_filter in zip(unit_filters, input_unit_filters):

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

        (firing_rates_simple_1, firing_rates_simple_2, 
         _, _,
         _, _, 
         n_trials_1, n_trials_2,
         _, _,
         _, _,
         _, _) = data_loader.reformat_firing_rate_data(data_format=None)
        
        n_neurons_1 = firing_rates_simple_1[0].shape[-1]
        n_neurons_2 = firing_rates_simple_2[0].shape[-1]

        ## The number of neurons should be the same for both conditions (ex. fast and slow)
        assert n_neurons_1 == n_neurons_2
        n_neurons.append(n_neurons_1)

    
    print(f'{session_date} {subject_id} {unit_filters} {task} {trial_filters} {n_trials_1}/{n_trials_2} {n_neurons}')



if __name__ == '__main__':

    # Iterate through every configured session so we can paste the summary table
    for session_data_name in session_data_dict.keys():
        generate_session_statistics(
            session_data_name)
        


# 20190412 N1 MC             CenterStart           fast, slow 58/58   134
# 20190517 N1 MC             CenterStart           fast, slow 58/71   127
# 20190528 N1 MC             CenterStart           fast, slow 64/78   146
# 20240516 N2 MC-LAT, MC-MED CenterStart           fast, slow 55/64   63, 78
# 20240530 N2 MC-LAT, MC-MED CenterStart           fast, slow 50/64   88, 43
# 20240816 N2 MC-LAT, MC-MED CenterStart           fast, slow 112/96  90, 79
# 20240820 N2 MC-LAT, MC-MED CenterStart           fast, slow 64/64   85, 82
# 20241015 N2 MC-LAT, MC-MED CenterStart           fast, slow 103/96  88, 101
# 20241022 N2 MC-LAT, MC-MED CenterStart           fast, slow 89/96   91, 75
# 20241105 N2 MC-LAT, MC-MED RadialGrid            near, far  93/92   103, 51
# 20241211 N2 MC-LAT, MC-MED RadialGrid            near, far  76/78   91, 52
# 20250408 N2 MC-LAT, MC-MED RadialGrid            near, far  113/120 73, 45
# 20250417 N2 MC-LAT, MC-MED CenterStartInterleave fast, slow 189/152 71, 70
# 20250422 N2 MC-LAT, MC-MED CenterStartInterleave fast, slow 175/112 61, 86
# 20250509 N2 MC-LAT, MC-MED CenterStartInterleave fast, slow 164/144 67, 76


# N2 20240516 CO
# N2 20240516 CO
# N2 20240530 CO
# N2 20240530 CO
# N2 20240816 CO
# N2 20240816 CO
# N2 20240820 CO
# N2 20240820 CO
# N2 20241015 CO
# N2 20241015 CO
# N2 20241022 CO
# N2 20241022 CO
# N2 20241105 RG
# N2 20241105 RG
# N2 20241211 RG
# N2 20241211 RG
# N2 20250408 RG
# N2 20250408 RG
# N2 20250417 COI
# N2 20250417 COI
# N2 20250422 COI
# N2 20250422 COI
# N2 20250509 COI
# N2 20250509 COI
