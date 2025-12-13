"""Utility to map deselected trial indices to per-run IDs for QC reports."""

import os
import numpy as np
import pandas as pd

import scripts.config as config



## Read parameters from config
data_dir = config.data_dir



def convert_deselected_trial_ids(
    session_data_name,
    deselected_trial_ids):
    """Group deselected trial indices by experimental run for easier logging."""

    ## Load trials data
    data_path_prefix = os.path.join(data_dir, session_data_name)
    trials_df = pd.read_csv(data_path_prefix + '_trials.csv')
    
    ## Filter out selected trials
    trials_df_deselected = trials_df.loc[deselected_trial_ids]
    n_runs = np.max(trials_df.Run) + 1

    ## Initialize list to store deselected trial ids by run
    deselected_trial_ids_by_run = []
    for i_run in range(n_runs):
        deselected_trial_ids_by_run.append(trials_df_deselected.loc[trials_df_deselected.Run == i_run].TrialNumber.values.tolist())

    print(deselected_trial_ids_by_run)



if __name__ == '__main__':

    deselected_trial_ids_dict = {
        # 'sub-N1_ses-20190412_CenterStart': [38, 54, 109, 110, 119, 126] + [8, 11, 73, 78, 79, 80], # fast and slow 1.5 path efficency threshold + positive slope within 0.35 distance
        # 'sub-N1_ses-20190517_CenterStart': [4, 6, 7, 9, 20, 25, 26, 28, 114, 116, 122, 124, 125, 128, 138, 144, 146, 147, 150, 155, 156, 157] + [42, 51, 62, 71, 93, 96, 100, 103, 105], # fast and slow 1.5 path efficency threshold + positive slope within 0.35 distance
        # 'sub-N1_ses-20190528_CenterStart': [48, 51, 53, 54, 55, 60, 63, 64, 69, 74, 78, 81, 82, 91, 94, 147, 149, 156, 157, 158, 159, 161, 169, 170, 171, 172, 173, 178, 179, 180, 186, 189] + [9, 13, 19, 28, 35, 39, 44, 45, 99, 105, 110, 115, 116, 121, 132, 134, 137, 142], # fast 1.5 path efficency threshold + positive slope within 0.35 distance
        # 'sub-N2_ses-20240516_CenterStart': [4, 12, 21, 29, 70, 72, 85, 90, 92], # fast and slow 1.5 path efficency threshold + positive slope within 0.35 distance
        # 'sub-N2_ses-20240530_CenterStart': [2, 9, 12, 14, 18, 28, 66, 67, 81, 86, 87, 89, 92, 94], # fast and slow 1.5 path efficency threshold + positive slope within 0.35 distance
        # 'sub-N2_ses-20240816_CenterStart': [38, 39, 109, 112, 117, 172, 177, 178, 180, 181, 186, 187, 190, 202, 221], # fast and slow 1.5 path efficency threshold + positive slope within 0.35 distance
        # 'sub-N2_ses-20240820_CenterStart': [32, 35, 39, 40, 42, 44, 45, 53, 54, 55, 57, 62, 96, 97, 101, 107, 110, 111, 112, 113, 119, 128, 129, 132, 133, 136, 142, 149, 151, 152, 153, 156], # fast and slow 1.5 path efficency threshold + positive slope within 0.35 distance
        # 'sub-N2_ses-20241015_CenterStart': [0, 16, 24, 64, 66, 67, 78, 82, 85, 92, 98, 100, 105, 111, 113, 115, 116, 120, 124, 125, 165, 169, 175, 187, 191], # fast and slow 1.5 path efficency threshold + positive slope within 0.35 distance
        # 'sub-N2_ses-20241022_CenterStart': [2, 9, 25, 131, 144, 145, 146], # fast and slow 1.5 path efficency threshold + positive slope within 0.35 distance
        # 'sub-N2_ses-20241105_RadialGrid': [79, 174, 181] + [11, 117, 131, 176], # near and far 1.5 path efficency threshold + positive slope within 0.17 or 0.49 distance
        # 'sub-N2_ses-20241211_RadialGrid': [0, 111, 119, 131] + [14, 55], # near and far 1.5 path efficency threshold + positive slope within 0.17 or 0.49 distance
        # 'sub-N2_ses-20250408_RadialGrid': [19, 29, 67, 139, 164, 201, 210], # near and far 1.5 path efficency threshold + positive slope within 0.17 or 0.49 distance
        'sub-N2_ses-20250417_CenterStartInterleave': [6, 23, 27, 28, 51, 65, 74, 90, 94, 96, 156, 177, 308, 309, 310, 318, 330, 344, 351], # fast 1.5 path efficency threshold + positive slope within 0.35 distance
        'sub-N2_ses-20250422_CenterStartInterleave': [20, 128, 140, 150, 159, 167, 168, 185, 216, 217, 238, 244, 247, 252, 263, 277, 296], # fast 1.5 path efficency threshold + positive slope within 0.35 distance
        'sub-N2_ses-20250509_CenterStartInterleave': [14, 32, 34, 82, 102, 110, 159, 165, 220, 224, 244, 314], # fast 1.5 path efficency threshold + positive slope within 0.35 distance
    }

    # Loop through every session we care about and summarize the run-level IDs
    for session_data_name, deselected_trial_ids in deselected_trial_ids_dict.items():
        print(session_data_name)
        convert_deselected_trial_ids(
            session_data_name,
            deselected_trial_ids)
        