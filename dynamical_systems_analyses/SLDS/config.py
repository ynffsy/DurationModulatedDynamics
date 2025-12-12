import os
import numpy as np

from sys import platform

if platform == 'linux' or platform == 'linux2':
    lab_dir = '/home/ynffsy/Desktop/andersen_lab'
elif platform == 'darwin':
    lab_dir = '/Users/ynffsy/Documents/andersen_lab_local'
elif platform == 'win32':
    lab_dir = 'X:\\andersen_lab_files'
else:
    raise ValueError('Unknown OS')


overwrite_results = False

data_dir    = os.path.join(lab_dir, 'data/cg/processed/')
results_dir = os.path.join(lab_dir, 'results/dynamics_paper')
vis_dir     = os.path.join(lab_dir, 'visualizations/dynamics_paper')


session_data_dict = {
    'sub-N1_ses-20190412_tf_CenterStart' : {
        'unit_filters' : ['MC'],
        'input_unit_filters' : [None],
        'trial_filters' : ['fast', 'slow'],
        'peak_onset_time_markers' : {'MC': 0.139},
        'peak_onset_time': {
            'fast': 0.125,
            'slow': 0.134,
            'all': 0.132,
        },
        'peak_time': {
            'fast': 0.171,
            'slow': 0.195,
            'all': 0.177,
        },
    },
    # 'sub-N1_ses-20190517_tf_CenterStart' : {
    #     'unit_filters' : ['MC'],
    #     'input_unit_filters' : [None],
    #     'trial_filters' : ['fast', 'slow'],
    #     'peak_onset_time_markers' : {'MC': 0.139},
    #     'peak_onset_time': {
    #         'fast': 0.127,
    #         'slow': 0.148,
    #         'all': 0.138,
    #     },
    #     'peak_time': {
    #         'fast': 0.174,
    #         'slow': 0.196,
    #         'all': 0.185,
    #     },
    # },
    # 'sub-N1_ses-20190528_tf_CenterStart' : {
    #     'unit_filters' : ['MC'],
    #     'input_unit_filters' : [None],
    #     'trial_filters' : ['fast', 'slow'],
    #     'peak_onset_time_markers' : {'MC': 0.139},
    #     'peak_onset_time': {
    #         'fast': 0.146,
    #         'slow': 0.155,
    #         'all': 0.151,
    #     },
    #     'peak_time': {
    #         'fast': 0.191,
    #         'slow': 0.204,
    #         'all': 0.198,
    #     },
    # },
    # 'sub-N2_ses-20240516_tf_CenterStart' : {
    #     'unit_filters' : ['MC-LAT', 'MC-MED'],
    #     'input_unit_filters' : [None, None],
    #     'trial_filters' : ['fast', 'slow'],
    #     'peak_onset_time_markers' : {'MC-LAT': 0.290, 'MC-MED': 0.275},
    #     'peak_onset_time': {
    #         'fast': [0.267, 0.276],
    #         'slow': [0.273, 0.276],
    #         'all': [0.265, 0.269],
    #     },
    #     'peak_time': {
    #         'fast': [0.329, 0.303],
    #         'slow': [0.315, 0.311],
    #         'all': [0.320, 0.307],
    #     },
    # },
    # 'sub-N2_ses-20240530_tf_CenterStart' : {
    #     'unit_filters' : ['MC-LAT', 'MC-MED'],
    #     'input_unit_filters' : [None, None],
    #     'trial_filters' : ['fast', 'slow'],
    #     'peak_onset_time_markers' : {'MC-LAT': 0.290, 'MC-MED': 0.275},
    #     'peak_onset_time': {
    #         'fast': [0.298, 0.282],
    #         'slow': [0.339, 0.303],
    #         'all': [0.311, 0.280],
    #     },
    #     'peak_time': {
    #         'fast': [0.359, 0.318],
    #         'slow': [0.380, 0.320],
    #         'all': [0.368, 0.320],
    #     },
    # },
    # 'sub-N2_ses-20240816_tf_CenterStart' : {
    #     'unit_filters' : ['MC-LAT', 'MC-MED'],
    #     'input_unit_filters' : [None, None],
    #     'trial_filters' : ['fast', 'slow'],
    #     'peak_onset_time_markers' : {'MC-LAT': 0.290, 'MC-MED': 0.275},
    #     'peak_onset_time': {
    #         'fast': [0.271, 0.279],
    #         'slow': [0.290, 0.408],
    #         'all': [0.272, 0.309],
    #     },
    #     'peak_time': {
    #         'fast': [0.351, 0.309],
    #         'slow': [0.359, 0.436],
    #         'all': [0.353, 0.367],
    #     },
    # },
    # 'sub-N2_ses-20240820_tf_CenterStart' : {
    #     'unit_filters' : ['MC-LAT', 'MC-MED'],
    #     'input_unit_filters' : [None, None],
    #     'trial_filters' : ['fast', 'slow'],
    #     'peak_onset_time_markers' : {'MC-LAT': 0.290, 'MC-MED': 0.275},
    #     'peak_onset_time': {
    #         'fast': [0.273, 0.297],
    #         'slow': [0.312, 0.383],
    #         'all': [0.273, 0.306],
    #     },
    #     'peak_time': {
    #         'fast': [0.324, 0.326],
    #         'slow': [0.340, 0.390],
    #         'all': [0.328, 0.338],
    #     },
    # },
    # 'sub-N2_ses-20241015_tf_CenterStart' : {
    #     'unit_filters' : ['MC-LAT', 'MC-MED'],
    #     'input_unit_filters' : [None, None],
    #     'trial_filters' : ['fast', 'slow'],
    #     'peak_onset_time_markers' : {'MC-LAT': 0.290, 'MC-MED': 0.275},
    #     'peak_onset_time': {
    #         'fast': [0.298, 0.298],
    #         'slow': [0.320, 0.284],
    #         'all': [0.298, 0.282],
    #     },
    #     'peak_time': {
    #         'fast': [0.353, 0.337],
    #         'slow': [0.377, 0.322],
    #         'all': [0.367, 0.328],
    #     },
    # },
    # 'sub-N2_ses-20241022_tf_CenterStart' : {
    #     'unit_filters' : ['MC-LAT', 'MC-MED'],
    #     'input_unit_filters' : [None, None],
    #     'trial_filters' : ['fast', 'slow'],
    #     'peak_onset_time_markers' : {'MC-LAT': 0.290, 'MC-MED': 0.275},
    #     'peak_onset_time': {
    #         'fast': [0.327, 0.348],
    #         'slow': [0.342, 0.326],
    #         'all': [0.322, 0.320],
    #     },
    #     'peak_time': {
    #         'fast': [0.363, 0.356],
    #         'slow': [0.392, 0.348],
    #         'all': [0.377, 0.354],
    #     },
    # },
    # 'sub-N2_ses-20241105_tf_RadialGrid' : {
    #     'unit_filters' : ['MC-LAT', 'MC-MED'],
    #     'input_unit_filters' : [None, None],
    #     'trial_filters' : ['near', 'far'],
    #     'peak_onset_time_markers' : {'MC-LAT': 0.290, 'MC-MED': 0.275},
    #     'peak_onset_time': {
    #         'fast': [0., 0.],
    #         'slow': [0., 0.],
    #         'all': [0., 0.],
    #     },
    #     'peak_time': {
    #         'fast': [0., 0.],
    #         'slow': [0., 0.],
    #         'all': [0., 0.],
    #     },
    # },
    # 'sub-N2_ses-20241211_tf_RadialGrid' : {
    #     'unit_filters' : ['MC-LAT', 'MC-MED'],
    #     'input_unit_filters' : [None, None],
    #     'trial_filters' : ['near', 'far'],
    #     'peak_onset_time_markers' : {'MC-LAT': 0.290, 'MC-MED': 0.275},
    #     'peak_onset_time': {
    #         'fast': [0., 0.],
    #         'slow': [0., 0.],
    #         'all': [0., 0.],
    #     },
    #     'peak_time': {
    #         'fast': [0., 0.],
    #         'slow': [0., 0.],
    #         'all': [0., 0.],
    #     },
    # },
    # 'sub-N2_ses-20250408_tf_RadialGrid' : {
    #     'unit_filters' : ['MC-LAT', 'MC-MED'],
    #     'input_unit_filters' : [None, None],
    #     'trial_filters' : ['near', 'far'],
    #     'peak_onset_time_markers' : {'MC-LAT': 0.290, 'MC-MED': 0.275},
    #     'peak_onset_time': {
    #         'fast': [0., 0.],
    #         'slow': [0., 0.],
    #         'all': [0., 0.],
    #     },
    #     'peak_time': {
    #         'fast': [0., 0.],
    #         'slow': [0., 0.],
    #         'all': [0., 0.],
    #     },
    # },
    # 'sub-N2_ses-20250417_tf_CenterStartInterleave' : {
    #     'unit_filters' : ['MC-LAT', 'MC-MED'],
    #     'input_unit_filters' : [None, None],
    #     'trial_filters' : ['fast', 'slow'],
    #     'peak_onset_time_markers' : {'MC-LAT': 0.290, 'MC-MED': 0.275},
    #     'peak_onset_time': {
    #         'fast': [0., 0.],
    #         'slow': [0., 0.],
    #         'all': [0., 0.],
    #     },
    #     'peak_time': {
    #         'fast': [0., 0.],
    #         'slow': [0., 0.],
    #         'all': [0., 0.],
    #     },
    # },
    # 'sub-N2_ses-20250422_tf_CenterStartInterleave' : {
    #     'unit_filters' : ['MC-LAT', 'MC-MED'],
    #     'input_unit_filters' : [None, None],
    #     'trial_filters' : ['fast', 'slow'],
    #     'peak_onset_time_markers' : {'MC-LAT': 0.290, 'MC-MED': 0.275},
    #     'peak_onset_time': {
    #         'fast': [0., 0.],
    #         'slow': [0., 0.],
    #         'all': [0., 0.],
    #     },
    #     'peak_time': {
    #         'fast': [0., 0.],
    #         'slow': [0., 0.],
    #         'all': [0., 0.],
    #     },
    # },
    # 'sub-N2_ses-20250509_tf_CenterStartInterleave' : {
    #     'unit_filters' : ['MC-LAT', 'MC-MED'],
    #     'input_unit_filters' : [None, None],
    #     'trial_filters' : ['fast', 'slow'],
    #     'peak_onset_time_markers' : {'MC-LAT': 0.290, 'MC-MED': 0.275},
    #     'peak_onset_time': {
    #         'fast': [0., 0.],
    #         'slow': [0., 0.],
    #         'all': [0., 0.],
    #     },
    #     'peak_time': {
    #         'fast': [0., 0.],
    #         'slow': [0., 0.],
    #         'all': [0., 0.],
    #     },
    # },
}



session_data_names = [
    # 'sub-N1_ses-20190412_tf_CenterStart',
    'sub-N1_ses-20190517_tf_CenterStart',
    'sub-N1_ses-20190528_tf_CenterStart',

    # 'sub-N2_ses-20240516_tf_CenterStart',
    # 'sub-N2_ses-20240530_tf_CenterStart',
    # 'sub-N2_ses-20240816_tf_CenterStart',
    # 'sub-N2_ses-20240820_tf_CenterStart',
    # 'sub-N2_ses-20241015_tf_CenterStart',
    # 'sub-N2_ses-20241022_tf_CenterStart',
    # 'sub-N2_ses-20250417_tf_CenterStartInterleave',
    # 'sub-N2_ses-20250422_tf_CenterStartInterleave',
    # 'sub-N2_ses-20250509_tf_CenterStartInterleave',
    # 'sub-N2_ses-20241105_tf_RadialGrid',
    # 'sub-N2_ses-20241211_tf_RadialGrid',
    # 'sub-N2_ses-20250408_tf_RadialGrid',

    # 'sub-N1_ses-20190412_CenterStart',
    # 'sub-N1_ses-20190517_CenterStart',
    # 'sub-N1_ses-20190528_CenterStart',

    # 'sub-N2_ses-20240516_CenterStart',
    # 'sub-N2_ses-20240530_CenterStart',
    # 'sub-N2_ses-20240816_CenterStart',
    # 'sub-N2_ses-20240820_CenterStart',
    # 'sub-N2_ses-20241015_CenterStart',
    # 'sub-N2_ses-20241022_CenterStart',
    # 'sub-N2_ses-20250417_CenterStartInterleave',
    # 'sub-N2_ses-20250422_CenterStartInterleave',
    # 'sub-N2_ses-20250509_CenterStartInterleave',
    # 'sub-N2_ses-20241105_RadialGrid',
    # 'sub-N2_ses-20241211_RadialGrid',

]

# unit_filter = 
unit_filters = [
    'MC',
    # 'MC-LAT',
    # 'MC-MED',
    # 'PPC',
    # 'PPC-IPL',
    # 'PPC-SPL',
]

input_unit_filters = [
    None,
    # 'PPC',
    # 'PPC-IPL',
    # 'PPC-SPL',
]

# window_config = 'gt_-0.2_fct_0.5_s0.001_gaussian_0.03_10'
# window_config = 'gt_0.0_fct_0.0_s0.01_gaussian_0.1_10'
window_config = 'gt_0.139_fct_0.0_s0.01_gaussian_0.1_10'
# window_config = 'gt_0.290_fct_0.0_s0.01_gaussian_0.1_10'
# window_config = 'gt_0.275_fct_0.0_s0.01_gaussian_0.1_10'
# window_config = 'gt_-0.2_fct_0.0_s0.01_gaussian_0.1_10'

# window_config = 'gt_0.15_fct_0.5_s0.02_gaussian_0.1_10'
# window_config = 'gt_-0.2_fct_0.0_s0.02_gaussian_0.03_10'
# window_config = 'gt_0.0_fct_0.0_s0.02_gaussian_0.1_10'
# window_config = 'gt_0.0_fct_0.0_s0.02_gaussian_0.03_10'
# window_config = 'gt_-0.2_fct_0.5_s0.02_gaussian_0.03_10'
# window_config = 'gt_0.1_fct_0.0_s0.001_gaussian_0.03_10'
# window_config = 'gt_0.0_fct_0.0_s0.001_gaussian_0.03_10'
# window_config = 'gt_0.0_fct_0.5_s0.02_gaussian_0.1_10'
# window_config = 'gt_0.1_fct_0.0_s0.02_gaussian_0.1_10'
# window_config = 'gt_0.15_fct_0.0_s0.02_gaussian_0.1_10'
# window_config = 'gt_0.283_fct_0.0_s0.01_gaussian_0.1_10'
# window_config = 'gt_0.283_fct_0.0_s0.02_gaussian_0.1_10'
# window_config = 'gt_0.286_fct_0.0_s0.02_gaussian_0.1_10'
# window_config = 'gt_0.290_fct_0.0_s0.02_gaussian_0.1_10'
# window_config = 'gt_0.294_fct_0.0_s0.02_gaussian_0.1_10'
# window_config = 'gt_0.25_fct_0.0_s0.02_gaussian_0.1_10'
# window_config = 'gt_0.28_fct_0.0_s0.02_gaussian_0.1_10'
# window_config = 'gt_0.3_fct_0.0_s0.02_gaussian_0.1_10'
# window_config = 'gt_0.3_fct_0.0_s0.02_gaussian_0.03_10'

time_step     = 0.01 ## in seconds
time_offset   = 0 ## index offset between MC and PPC

data_formats  = [None]  ## None, 'concat', 'fill0', 'truncate_end', 'truncate_front' or 'resample'
label_formats = ['pe'] ## 'cartesian', 'univec', 'pe'
standardize   = False

trial_filters = [
    'fast',
    'slow',
    # 'near',
    # 'far',
    # 'masked_far',
    # 'masked_near',
    # 'unmasked_far',
    # 'unmasked_near',
    # 'masked',
    # 'unmasked',
    # None,
]

train_test_options = [ # 'same_speed' or 'cross_speed'
    'same_speed', 
    # 'cross_speed',
    # 'joint',
] 


# random_states = [0, 9, 42, 66, 99]
# random_states = [99]
random_states = [42]
n_folds       = 5
# ns_states     = [2, 3, 4] + np.arange(5, 81, 5).tolist()
# ns_states     = [2, 3, 4] + np.arange(5, 21, 5).tolist()
# ns_states     = [2, 3, 4] + np.arange(5, 11, 5).tolist()
# ns_states     = np.arange(2, 21).tolist()
# ns_states     = [40, 50, 60]
# ns_states     = [20]
# ns_states     = [2, 3]
ns_states     = [3]
# ns_discrete_states = [2, 3, 4, 5, 6, 7, 8, 9, 10, 16]
ns_discrete_states = [1, 2, 3, 4, 6, 8, 12, 16]
# ns_discrete_states = [1, 2, 3]
# ns_discrete_states = [2]
# ns_discrete_states = [16]
# ns_discrete_states = [1]
# ns_iters      = [1] + np.arange(10, 101, 10).tolist()  
# ns_iters      = [1] + np.arange(10, 1001, 50).tolist()
# ns_iters      = [100]
ns_iters      = [25]

## window_sizes is used for the baseline case of inference
window_sizes  = [2, 3, 4, 5, 6, 7, 8, 9, 10]

model_types        = ['rSLDS'] # 'LDS' or 'rSLDS'
dynamics_classes   = ['diagonal_gaussian']
emission_classes   = ['gaussian']
init_types         = ['ARHMM'] # 'ARHMM', or None
subspace_types     = [None]
alphas             = [0.5]
# alphas             = [0.0, 1.0]
