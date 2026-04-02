"""Central configuration shared across all SLDS analysis scripts.

This module centralizes directory paths, session metadata, trial filters,
and model sweep parameters so that different training scripts can import a
single data source. Adjust values here to change experiment settings without 
touching the execution code.
"""

import os
import numpy as np


## ── Root directory ──────────────────────────────────────────────────
## Set the environment variable DMD_BASE_DIR to point to the directory
## that contains the data/, results/, and visualizations/ sub-folders.
##
##   export DMD_BASE_DIR=/path/to/your/base/directory
##
## Alternatively, create a file called .env in the repository root with:
##   DMD_BASE_DIR=/path/to/your/base/directory
## ────────────────────────────────────────────────────────────────────
_base_dir = os.environ.get('DMD_BASE_DIR')
if _base_dir is None:
    # Try loading from a .env file at the repo root
    _env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.isfile(_env_path):
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith('#') and '=' in _line:
                    _key, _val = _line.split('=', 1)
                    if _key.strip() == 'DMD_BASE_DIR':
                        _base_dir = _val.strip()
                        break
if _base_dir is None:
    raise RuntimeError(
        'DMD_BASE_DIR is not set. Either:\n'
        '  1. export DMD_BASE_DIR=/path/to/base/dir   (in your shell), or\n'
        '  2. create a .env file in the repo root with DMD_BASE_DIR=/path/to/base/dir'
    )


## Global flags shared by every training job
overwrite_results = False

## Canonical locations for neural data, intermediate outputs, and figures
data_dir    = os.path.join(_base_dir, 'data')
results_dir = os.path.join(_base_dir, 'results')
vis_dir     = os.path.join(_base_dir, 'visualizations')

## List of session identifiers to include in the current sweep
session_data_names = [
    'sub-N1_ses-20190412_tf_CenterStart',
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
]

## Which neural populations to keep when loading data
# unit_filter = 
unit_filters = [
    'MC',
    # 'MC-LAT',
    # 'MC-MED',
    # 'PPC',
    # 'PPC-IPL',
    # 'PPC-SPL',
]

## Optional secondary population used as exogenous inputs
input_unit_filters = [
    None,
    # 'PPC',
    # 'PPC-IPL',
    # 'PPC-SPL',
]

## Temporal preprocessing window configuration applied before fitting models
# window_config = 'gt_-0.2_fct_0.5_s0.001_gaussian_0.03_10'
# window_config = 'gt_0.0_fct_0.0_s0.01_gaussian_0.1_10'
window_config = 'gt_0.132_fct_0.0_s0.01_gaussian_0.1_10' # N1 MC
# window_config = 'gt_0.275_fct_0.0_s0.01_gaussian_0.1_10' # N2 MC-LAT
# window_config = 'gt_0.248_fct_0.0_s0.01_gaussian_0.1_10' # N2 MC-MED
# window_config = 'gt_-0.2_fct_0.0_s0.01_gaussian_0.1_10'


## Sampling parameters shared by decoding/visualization code paths
time_step     = 0.01 ## in seconds
time_offset   = 0 ## index offset between MC and PPC

## Possible tensor reshaping strategies and label conventions
data_formats  = [None]  ## None, 'concat', 'fill0', 'truncate_end', 'truncate_front' or 'resample'
label_formats = ['pe'] ## 'cartesian', 'univec', 'pe'
standardize   = False

## Each top-level script iterates over these behavioral conditions
trial_filters = [
    'fast',
    'slow',
    # 'near',
    # 'far',
]

## Select which training/evaluation paradigms to run
train_test_options = [
    'same_speed', 
    # 'cross_speed',
    # 'joint',
] 


## Shared sweep definitions for randomness, CV depth, and model sizes
random_states = [42]
n_folds       = 5
# ns_states     = np.arange(2, 21).tolist()
# ns_states     = [2, 3]
ns_states     = [3]
ns_discrete_states = [1, 2, 3, 4, 6, 8, 12, 16]
ns_iters      = [25]

## window_sizes is used for the baseline case of inference
window_sizes  = [2, 3, 4, 5, 6, 7, 8, 9, 10]

## Model family definitions used by the sweeps
model_types        = ['rSLDS']
dynamics_classes   = ['diagonal_gaussian']
emission_classes   = ['gaussian']
init_types         = ['ARHMM'] # 'ARHMM', or None
subspace_types     = [None]
alphas             = [0.5]
