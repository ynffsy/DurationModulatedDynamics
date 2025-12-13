"""Shared data wrangling helpers for loading, filtering, and aligning trials."""

import os
import ipdb
import pickle
import numpy as np
import pandas as pd
import xarray as xr

from scipy.interpolate import interp1d
from scipy.spatial.distance import squareform
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm_crossnobis
# from rsatoolbox.inference import evaluate_models
from rsatoolbox.model import ModelFixed
from rsatoolbox.rdm import calc_rdm
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import rsatoolbox



trial_filter_counterparts = {
    'slow'         : 'fast',
    'fast'         : 'slow',
    'far'          : 'near',
    'near'         : 'far',
    'masked_far'   : 'masked_near',
    'masked_near'  : 'masked_far',
    'unmasked_far' : 'unmasked_near',
    'unmasked_near': 'unmasked_far',
    'masked'       : 'unmasked',
    'unmasked'     : 'masked',
}

window_config_to_end_adjustment = {
    'movement_duration': 0.0,
    'base_go'          : 0.0,
    'extended_go'      : 0.5,
}



def split_by_lengths(scores_flat, lengths):
    """Return a list of (T_i, n_components) arrays."""
    out, idx = [], 0
    for L in lengths:
        out.append(scores_flat[idx : idx+L, :])
        idx += L
    return out


def min_list_of_2d_np(data, last_dim=None):
    if last_dim is not None:
        return np.min([np.min(data_[:, last_dim]) for data_ in data])
    else:
        return np.min([np.min(data_) for data_ in data])


def max_list_of_2d_np(data, last_dim=None):
    if last_dim is not None:
        return np.max([np.max(data_[:, last_dim]) for data_ in data])
    else:
        return np.max([np.max(data_) for data_ in data])


## NOTE: This is slightly different from the MATLAB trajectory straightness due to data range.
def compute_trajectory_straightness(cursor_pos, target_pos):

    n_trials = cursor_pos.shape[0]
    straightness_all = []

    for i in range(n_trials):
            
        cursor_pos_ = cursor_pos[i, :, :]
        target_pos_ = target_pos[i, :]

        cursor_pos_angles = np.degrees(np.arctan2(cursor_pos_[:, 1], cursor_pos_[:, 0]))
        target_pos_angle  = np.degrees(np.arctan2(target_pos_[1],    target_pos_[0]))

        cursor_pos_angle_diffs = np.abs((cursor_pos_angles - target_pos_angle + 180) % 360 - 180)

        straightness_all.append(np.mean(cursor_pos_angle_diffs))

    return np.array(straightness_all)


def compute_trajectory_path_efficiency(cursor_pos, target_pos):
    """
    Compute path-efficiency on a per-trial basis.

    For each trial we treat the *effective* target as the point on the
    line from the start position to the nominal target centre that is
    closest (orthogonal projection) to the last cursor sample.  
    This provides a fair baseline even when we keep only one sample that
    overlaps the target.

    Parameters
    ----------
    cursor_pos  : array_like, shape (n_trials, n_samples, 2)
                  XY cursor positions for every trial.
    target_pos  : array_like, shape (n_trials, 2)
                  XY centre of the target for every trial.

    Returns
    -------
    path_efficiencies : ndarray, shape (n_trials,)
        travelled-path / ideal-straight-line  (values ≥ 1 indicate extra
        path length relative to the straight-line baseline).
    """
    cursor_pos   = np.asarray(cursor_pos)
    target_pos   = np.asarray(target_pos)
    n_trials     = cursor_pos.shape[0]
    efficiencies = np.empty(n_trials, dtype=float)

    for i in range(n_trials):
        xy        = cursor_pos[i]          # (T, 2)
        start     = xy[0]                  # starting position
        finish    = xy[-1]                 # last recorded sample
        centre    = target_pos[i]

        # --- project the finish point onto the start→centre line -----------
        v         = centre - start         # direction to nominal target
        v_norm2   = np.dot(v, v)
        # if start == centre, v_norm2 == 0 → treat ideal dist as ε
        if v_norm2 == 0.0:
            ideal_dist = np.finfo(float).eps
        else:
            t           = np.dot(finish - start, v) / v_norm2
            t           = np.clip(t, 0.0, 1.0)      # keep within the segment
            eff_target  = start + t * v
            ideal_dist  = np.linalg.norm(eff_target - start)

        # --- actual distance travelled -------------------------------------
        travelled  = np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))

        efficiencies[i] = travelled / ideal_dist if ideal_dist > 0 else np.nan
        #  (nan only possible if start == centre, an ill-posed trial)

        # ipdb.set_trace()

    return efficiencies


def analyze_trajectory_slopes(cursor_pos, target_pos, start_distance):
    """
    Steepest (most-negative) Δ(distance)/Δt for each trial, measured only
    after the cursor first comes within `start_distance` of the target.

    Parameters
    ----------
    cursor_pos     : array_like, shape (n_trials, n_samples, 2)
        XY cursor positions for every trial.
    target_pos     : array_like, shape (n_trials, 2)
        XY centre of the target for every trial.
    start_distance : float
        Begin analysing slopes as soon as the cursor–target distance is
        ≤ `start_distance`.  If that never occurs, the result is NaN.

    Returns
    -------
    max_slopes : ndarray, shape (n_trials,)
        The most-negative slope (px / frame) recorded after the threshold
        is crossed.  `np.nan` if the threshold is never reached or if
        there is only a single sample in that range.
    """
    cursor_pos = np.asarray(cursor_pos)
    target_pos = np.asarray(target_pos)

    n_trials   = cursor_pos.shape[0]
    max_slopes = np.empty(n_trials, dtype=float)
    ns_positive_slopes = np.zeros(n_trials, dtype=int)

    for i in range(n_trials):
        xy  = cursor_pos[i]          # (T, 2)
        tgt = target_pos[i]

        # Distance to target at each time step
        dist = np.linalg.norm(xy - tgt, axis=1)

        # Index where we first get within the specified distance
        inside_idx = np.where(dist <= start_distance)[0]
        if inside_idx.size == 0 or inside_idx[0] >= dist.size - 1:
            # Threshold never reached, or not enough samples to form a slope
            max_slopes[i] = np.nan
            continue

        first_in      = inside_idx[0]
        slopes        = np.diff(dist)               # Δ(distance) per frame
        slopes_window = slopes[first_in:]           # start counting here

        max_slopes[i] = np.min(slopes_window)       # most-negative value
        ns_positive_slopes[i] = np.sum(slopes_window > 0)

    return max_slopes, ns_positive_slopes


def extract_trial_firing_rates(firing_rates, trial_number_filter=None, cutoff_times=None):

    firing_rates_new    = []
    firing_rates_simple = []
    trial_ids           = []

    if cutoff_times is not None:
        assert len(cutoff_times) == len(firing_rates)

    for i in range(len(firing_rates)):

        if trial_number_filter is not None and i in trial_number_filter:
            continue

        if cutoff_times is not None:
            if cutoff_times[i] is None:
                continue
            
            ## Extract the firing rates up to the cutoff time
            firing_rate_times = np.array(firing_rates[i].time)
            max_firing_rate_time_idx = np.argmax(firing_rate_times[firing_rate_times < cutoff_times[i]])
            trimmed_firing_rates = firing_rates[i].data[:max_firing_rate_time_idx, :]
            firing_rates_simple.append(trimmed_firing_rates)

            ## Create a new DataArray with updated data and time
            firing_rates_ = xr.DataArray(
                trimmed_firing_rates,
                dims=['time', 'unit_id'],
                coords={'time': firing_rate_times[:max_firing_rate_time_idx], 'unit_id': firing_rates[i].coords['unit_id']},
                name='firing rate'
            )

            ## Copy over the attributes
            firing_rates_.attrs = firing_rates[i].attrs

            ## Save the new DataArray
            firing_rates_new.append(firing_rates_)

        else:
            firing_rates_new.append(firing_rates[i])
            firing_rates_simple.append(firing_rates[i].data)

        ## Add the trial id
        trial_ids.append(firing_rates[i].trial_id)

    return firing_rates_new, firing_rates_simple, np.array(trial_ids)


def extract_cursor_states(
    cursor_pos_timeseries, 
    cursor_vel_timeseries, 
    trial_ids):

    cursor_states_all = []

    for trial_id in trial_ids:
        
        cursor_pos = np.array(cursor_pos_timeseries[trial_id])
        cursor_vel = np.array(cursor_vel_timeseries[trial_id])

        cursor_states = np.column_stack((cursor_pos, cursor_vel))
        cursor_states_all.append(cursor_states)

    return cursor_states_all


def align_cursor_to_fr_times(
    firing_rates_all, 
    cursor_pos_timeseries, 
    cursor_vel_timeseries,
    trial_ids,
    times_new=None):
    """
    Align cursor times to firing rate times for each trial using interpolation 
        and extrapolation.
    """
    n_trials = len(firing_rates_all)
    assert n_trials == len(cursor_pos_timeseries)
    assert n_trials == len(cursor_vel_timeseries)

    cursor_states_all = []

    for i, trial_id in enumerate(trial_ids):

        if times_new is not None:
            times_fr = times_new[i]
        else:
            times_fr = firing_rates_all[i].time

        times_cursor = cursor_pos_timeseries[trial_id].time
        assert np.all(times_cursor == cursor_vel_timeseries[trial_id].time)

        f_pos_x = interp1d(times_cursor, cursor_pos_timeseries[trial_id][:, 0], 
                           kind='linear', fill_value='extrapolate')
        f_pos_y = interp1d(times_cursor, cursor_pos_timeseries[trial_id][:, 1], 
                           kind='linear', fill_value='extrapolate')
        f_vel_x = interp1d(times_cursor, cursor_vel_timeseries[trial_id][:, 0], 
                           kind='linear', fill_value='extrapolate')
        f_vel_y = interp1d(times_cursor, cursor_vel_timeseries[trial_id][:, 1], 
                           kind='linear', fill_value='extrapolate')
        
        pos_x_aligned = f_pos_x(times_fr)
        pos_y_aligned = f_pos_y(times_fr)
        vel_x_aligned = f_vel_x(times_fr)
        vel_y_aligned = f_vel_y(times_fr)

        cursor_states = np.column_stack((
            pos_x_aligned, pos_y_aligned, vel_x_aligned, vel_y_aligned))

        cursor_states_all.append(cursor_states)

    return cursor_states_all


def fill_emissions(emissions):

    n_trials    = len(emissions)
    n_neurons   = emissions[0].shape[1]
    max_n_times = 0
    
    n_times_all = []

    ## Find the trial with the most time steps and the number of time steps for each trial
    for i in range(n_trials):
        n_times = emissions[i].shape[0]
        n_times_all.append(n_times)
        max_n_times = max(max_n_times, n_times)

    ## Create a new emissions array with the same number of time steps for all trials. 
    ## Fill the extra time steps with zeros.
    emissions_new = np.zeros((n_trials, max_n_times, n_neurons))
    # emissions_new = np.full((n_trials, max_n_times, n_neurons), np.nan)

    for i, n_times in enumerate(n_times_all):
        emissions_new[i, :n_times, :] = emissions[i]
    
    return emissions_new, np.array(n_times_all)


def filter_by_trial_length_percentile(emissions, percentile, *, make_copies=True):
    """
    Keep trials whose length is at or above the (100 − percentile)th percentile.

    Parameters
    ----------
    emissions   : list of (T_i, N) ndarrays
    percentile  : 0 < p ≤ 100 → drop the shortest p % of trials.
                  If None, return everything unchanged.
    make_copies : If True, return copies of the kept arrays so that later
                  in-place edits won’t touch the originals.
    """
    trial_lengths = np.fromiter((tr.shape[0] for tr in emissions), int)

    if percentile is None:                       # no filtering requested
        keep_mask = np.ones_like(trial_lengths, dtype=bool)
        return emissions, trial_lengths, keep_mask

    if not (0 < percentile <= 100):
        raise ValueError("percentile must be in (0, 100].")

    cutoff    = np.percentile(trial_lengths, 100 - percentile)
    keep_mask = trial_lengths >= cutoff
    if not keep_mask.any():
        raise ValueError("Percentile threshold removed every trial!")

    if make_copies:
        emissions_filtered = [tr.copy() for tr, k in zip(emissions, keep_mask) if k]
    else:
        emissions_filtered = [tr for tr, k in zip(emissions, keep_mask) if k]

    return emissions_filtered, trial_lengths[keep_mask], keep_mask


def truncate_emissions(
    emissions,
    truncate_end=True,
):
    """
    Parameters
    ----------
    emissions : list[array_like]
        List of (T_i, N) arrays, one per trial.
    truncate_end : bool, default True
        If True, cut from the end; otherwise cut from the start.
    """
    n_trials_total = len(emissions)
    n_neurons = emissions[0].shape[1]

    # -------- 1. gather trial lengths --------
    trial_lengths = np.array([tr.shape[0] for tr in emissions])

    # -------- 3. truncate the kept trials --------
    min_len = int(trial_lengths.min())
    n_trials_kept = len(emissions)

    emissions_new = np.zeros((n_trials_kept, min_len, n_neurons))
    times_new_all = []

    for i, tr in enumerate(emissions):
        if truncate_end:
            emissions_new[i] = tr[:min_len]
            if hasattr(tr, 'time'):
                times_new_all.append(tr.time[:min_len])
        else:
            emissions_new[i] = tr[-min_len:]
            if hasattr(tr, 'time'):
                times_new_all.append(tr.time[-min_len:])

    return emissions_new, np.repeat(min_len, n_trials_kept), times_new_all


def truncate_discrete_states(discrete_states, truncate_end=True):

    n_trials = len(discrete_states)
    trial_lengths = []

    ## Extract trial lengths
    for i_trial in range(n_trials):
        trial_lengths.append(len(discrete_states[i_trial]))

    min_trial_length = min(trial_lengths)

    ## Create a array with the same trial length for all trials truncated either at the front or the end
    discrete_states_truncated = np.zeros((n_trials, min_trial_length))

    for i_trial in range(n_trials):
        if truncate_end:
            discrete_states_truncated[i_trial, :] = discrete_states[i_trial][:min_trial_length]
        else:
            discrete_states_truncated[i_trial, :] = discrete_states[i_trial][-min_trial_length:]
    
    return discrete_states_truncated


def resample_emissions(emissions, trial_length_new=None, trial_length_end_buffer=0, default_time_step=0.02):

    n_trials      = len(emissions)
    n_neurons     = emissions[0].shape[1]    
    trial_lengths = []

    ## Extract trial lengths
    for i_trial in range(n_trials):
        trial_length = emissions[i_trial].shape[0] - trial_length_end_buffer
        trial_lengths.append(trial_length)

    trial_length_median = int(np.median(trial_lengths))  ## Excluding the end buffer

    if trial_length_new is None:
        trial_length_new = trial_length_median
    
    trial_length_full = trial_length_new + trial_length_end_buffer

    ## Create a new emissions array with the same trial length for all trials
    ## Trials will be resampled to the median trial length
    emissions_new = np.zeros((n_trials, trial_length_full, n_neurons))
    times_new_all = []

    for i_trial, trial_length in enumerate(trial_lengths):

        try:
            times = emissions[i_trial].time
        except:
            times = np.arange(trial_length) * default_time_step
            
        time_start = times[0]
        time_end   = times[-1 - trial_length_end_buffer]

        times_base = np.linspace(time_start, time_end, trial_length_new)

        if trial_length_end_buffer > 0:
            times_full = np.append(times_base, times[-trial_length_end_buffer:])
        else:
            times_full = times_base

        times_new_all.append(times_full)

        for i_neuron in range(n_neurons):
            if trial_length_end_buffer > 0:
                ## Interpolate emissions up to the end buffer and append the last trial_length_end_buffer time steps
                emission_interp = interp1d(times[:-trial_length_end_buffer], emissions[i_trial][:-trial_length_end_buffer, i_neuron], kind='linear', fill_value='extrapolate')
                emissions_new[i_trial, :trial_length_new, i_neuron] = emission_interp(times_base)
                emissions_new[i_trial, trial_length_new:, i_neuron] = emissions[i_trial][-trial_length_end_buffer:, i_neuron]
                
            else:
                emission_interp = interp1d(times, emissions[i_trial][:, i_neuron], kind='linear', fill_value='extrapolate')
                emissions_new[i_trial, :, i_neuron] = emission_interp(times_base)

    return emissions_new, np.repeat(trial_length_full, n_trials), times_new_all


def resample_discrete_states(discrete_states, trial_lengths=None):

    n_trials = len(discrete_states)

    if trial_lengths is None:
        trial_lengths = []

        ## Extract trial lengths
        for i_trial in range(n_trials):
            trial_lengths.append(len(discrete_states[i_trial]))

    trial_length_median = int(np.median(trial_lengths))

    ## Create a new discrete state array with the same trial length for all trials
    ## Trials will be resampled to the median trial length
    discrete_states_new = np.zeros((n_trials, trial_length_median))

    for i_trial, trial_length in enumerate(trial_lengths):

        times = np.arange(trial_length)
            
        time_start = times[0]
        time_end   = times[-1]

        times_base = np.linspace(time_start, time_end, trial_length_median)

        discrete_states_interp = interp1d(times, discrete_states[i_trial], kind='linear', fill_value='extrapolate')
        discrete_states_new[i_trial, :] = discrete_states_interp(times_base)

    discrete_states_new = np.round(discrete_states_new).astype(np.int32)
    return discrete_states_new


def resample_times(times, trial_lengths=None):

    n_trials = len(times)

    if trial_lengths is None:
        trial_lengths = []

        ## Extract trial lengths
        for i_trial in range(n_trials):
            trial_lengths.append(len(times[i_trial]))

    trial_length_median = int(np.median(trial_lengths))

    ## Create a new times array with the same trial length for all trials
    ## Trials will be resampled to the median trial length
    times_new = np.zeros((n_trials, trial_length_median))

    for i_trial, times_ in enumerate(times):
            
        times_min = times_[0]
        times_max = times_[-1]

        times_resampled = np.linspace(times_min, times_max, trial_length_median)
        times_new[i_trial, :] = times_resampled

    return times_new


def convert_states_to_polar_simple(cursor_states):
    """
    Expects cursor_states to be an array of length n_trials, each element of which with shape (n_times, 4) 
    """

    cursor_states_new = np.zeros_like(cursor_states).astype(float)

    x_pos = cursor_states[:, 0]
    y_pos = cursor_states[:, 1]
    x_vel = cursor_states[:, 2]
    y_vel = cursor_states[:, 3]

    mag_pos = np.sqrt(x_pos**2 + y_pos**2)
    ang_pos = np.degrees(np.arctan2(y_pos, x_pos))

    mag_vel = np.sqrt(x_vel**2 + y_vel**2)
    ang_vel = np.degrees(np.arctan2(y_vel, x_vel))

    cursor_states_new[:, 0] = mag_pos
    cursor_states_new[:, 1] = ang_pos
    cursor_states_new[:, 2] = mag_vel
    cursor_states_new[:, 3] = ang_vel

    return cursor_states_new


def convert_states_to_polar(cursor_states):
    """
    Expects cursor_states to be an array of length n_trials, each element of which with shape (n_times, 4) 
    """

    cursor_states_new = []

    for cursor_states_ in cursor_states:

        cursor_states_new_ = convert_states_to_polar_simple(cursor_states_)
        cursor_states_new.append(cursor_states_new_)

    return cursor_states_new


def compute_direction_to_target(cursor_states, target_positions, univec=False):
    """
    Expects cursor_states to be an array of length n_trials, each element of which with shape (n_times, 4) 
    Expects target_positions to be an array with shape (n_trials, 2)
    """

    assert len(cursor_states) == target_positions.shape[0]

    direction_to_target_all = []

    for i, cursor_states_ in enumerate(cursor_states):

        cursor_pos = cursor_states_[:, [0, 1]]
        target_pos = target_positions[i, :]

        ## Compute the direction to the target with respect to the cursor position
        direction_to_target = target_pos - cursor_pos

        if univec:
            direction_to_target /= np.linalg.norm(direction_to_target, axis=1)[:, None]

        direction_to_target_all.append(direction_to_target)

    return direction_to_target_all


def compute_crossnobis_matrix(
    firing_rates_simple, 
    time_step=None):

    n_trials, n_times, n_neurons = firing_rates_simple.shape

    if time_step is not None:
        firing_rates_simple = firing_rates_simple / time_step

    ## Create a TemporalDataset object for the firing rates
    ## https://rsatoolbox.readthedocs.io/en/latest/datasets.html
    ## TemporalDataset expects shape (n_trials, n_neurons, n_times)
    temporal_dataset = rsatoolbox.data.TemporalDataset(
        np.transpose(firing_rates_simple, (0, 2, 1)),  
        channel_descriptors={'neuron': np.arange(n_neurons)},
        # obs_descriptors={'stimulus': stimulus},
        time_descriptors={'time': np.arange(n_times)}
    )

    ## Convert TemporalDataset to Dataset
    dataset = temporal_dataset.convert_to_dataset(by='time')

    ## Compute the crossnobis matrix
    results, rdm_vectors = calc_rdm_crossnobis(dataset, descriptor='time')

    ## Get the averaged crossnobis matrix
    crossnobis_matrix = results[0].get_matrices()[0]

    ## Restore rdms_all to the same format (it was stored as upper triangular matrices)
    n_trials = rdm_vectors.shape[0]
    n_cond = crossnobis_matrix.shape[0]

    crossnobis_matrices = np.ndarray((n_trials, n_cond, n_cond))
    for i_trial in np.arange(n_trials):
        crossnobis_matrices[i_trial, :, :] = squareform(rdm_vectors[i_trial, :])  # This is the same implementation as in rsatoolbox

    # -----------------------------------------------------
    # SHAPE consistency
    assert crossnobis_matrix.shape[0] == crossnobis_matrix.shape[1], \
        "crossnobis_matrix must be square"
    assert crossnobis_matrices.shape[1:] == crossnobis_matrix.shape, \
        "per-fold matrices must match the grand matrix shape"

    # -----------------------------------------------------
    # SYMMETRY  (diagonal zero, off-diagonal mirrored)
    eps = 1e-10
    assert np.allclose(np.diag(crossnobis_matrix), 0, atol=eps)
    assert np.allclose(crossnobis_matrix, crossnobis_matrix.T, atol=eps)

    # each fold should be symmetric and diagonal-zero, too
    for m in crossnobis_matrices:
        assert np.allclose(m, m.T, atol=eps)
        assert np.allclose(np.diag(m), 0, atol=eps)

    # -----------------------------------------------------
    # GRAND average really equals the mean of folds?
    assert np.allclose(crossnobis_matrix,
                       crossnobis_matrices.mean(0), atol=eps), \
           "Grand matrix should equal mean of per-fold matrices"

    return crossnobis_matrix, crossnobis_matrices


def compute_crossnobis_matrix_manual(firing_rates_simple):


    num_trials, num_time_points, num_neurons = firing_rates_simple.shape

    mean_responses = np.mean(firing_rates_simple, axis=0)  # Shape: (num_time_points, num_neurons)
    cov_matrices = np.array([np.cov(firing_rates_simple[:, t, :], rowvar=False) for t in range(num_time_points)])  # Shape: (num_time_points, num_neurons, num_neurons)


    # from scipy.spatial.distance import mahalanobis

    crossnobis_distances = []

    for t in range(num_time_points - 1):
        mean_diff = mean_responses[t + 1] - mean_responses[t]
        pooled_cov = (cov_matrices[t] + cov_matrices[t + 1]) / 2
        inv_pooled_cov = np.linalg.inv(pooled_cov)

        distance = np.dot(mean_diff.T, np.dot(inv_pooled_cov, mean_diff))
        crossnobis_distances.append(distance)

    # Convert to numpy array for easier manipulation
    crossnobis_distances = np.array(crossnobis_distances)


    # import matplotlib.pyplot as plt

    time_points = np.arange(num_time_points - 1)  # Time points for distances

    plt.plot(time_points, crossnobis_distances)
    plt.xlabel('Time Points')
    plt.ylabel('Crossnobis Distance')
    plt.title('Evolution of Neural Distances Over Time')
    plt.show()


def compute_cross_speed_crossnobis_distance(
    continuous_states_1: np.ndarray,
    continuous_states_2: np.ndarray,
):
    """
    Parameters
    ----------
    continuous_states_1 : array (n_trials_1, 100, K)
    continuous_states_2 : array (n_trials_2, 100, K)

    Returns
    -------
    crossnobis : array (100,)
        Cross‑validated squared Mahalanobis distance between
        two latent trajectories at each warped time bin.
    """

    n_trials_1, trial_len, n_latent = continuous_states_1.shape
    n_trials_2 = continuous_states_2.shape[0]
    assert trial_len == continuous_states_2.shape[1]
    assert n_latent == continuous_states_2.shape[2]

    # ---------------------------------------------------------------------
    #  Stack trials from both speeds →  (n_trials, 100, K)
    # ---------------------------------------------------------------------
    latents_all   = np.vstack([continuous_states_1, continuous_states_2])
    cond_labels   = np.hstack([np.zeros(n_trials_1,  dtype=int),
                               np.ones (n_trials_2, dtype=int)])
    run_labels    = np.arange(latents_all.shape[0]) # each trial is its own “run”

    # ---------------------------------------------------------------------
    #  Compute cross‑nobis distance at every warped time point
    # ---------------------------------------------------------------------
    crossnobis = np.zeros(trial_len)
    for t in range(trial_len):

        # patterns:  (n_trials  ,  K)
        patterns = latents_all[:, t, :]

        # Build rsatoolbox Dataset
        ds = Dataset(
            patterns,
            obs_descriptors={
                "speed" : cond_labels,   # condition we want the RDM over
                "run"   : run_labels     # partitions for cross‑validation
            },
            descriptors={"time" : t}
        )

        # Compute 1×1 RDM (because only two conditions); extract the single value
        rdm, _ = calc_rdm_crossnobis(ds,
                       descriptor    = "speed",
                       cv_descriptor = "run")

        crossnobis[t] = rdm.dissimilarities[0]

    return crossnobis


def compute_cross_speed_crossnobis_matrix(states_1, states_2):

    n_trials_1, T, K = states_1.shape
    n_trials_2       = states_2.shape[0]

    # ------------------------------------------------------------
    # 0. Need ≥2 trials per speed so every (speed,time) condition
    #    appears in *both* cross‑validation folds.
    # ------------------------------------------------------------
    assert n_trials_1 >= 2 and n_trials_2 >= 2, \
        "cross‑nobis needs at least 2 trials per speed"

    # ------------------------------------------------------------
    # 1.  Stack trial × time  →  pattern rows
    # ------------------------------------------------------------
    #   patterns_all : (N, K)   N = (n_trials_1 + n_trials_2)*T
    #   For each trial we keep all T time bins consecutively.
    patterns_1 = states_1.reshape(n_trials_1 * T, K)
    patterns_2 = states_2.reshape(n_trials_2 * T, K)
    patterns_all  = np.vstack([patterns_1, patterns_2])

    # Condition coding  -----------------------------------------
    #   cond = speed * T + time        ∈ {0..199}  (if T=100)
    time_rep_1 = np.tile(np.arange(T), n_trials_1)
    time_rep_2 = np.tile(np.arange(T), n_trials_2)
    cond_labels   = np.concatenate([
        0 * T + time_rep_1,
        1 * T + time_rep_2])

    # Two cross‑validation folds  -------------------------------
    #   fold 0 = even‑indexed trials, fold 1 = odd‑indexed trials
    run_labels_1 = np.repeat(np.arange(n_trials_1) % 2, T)       # 0/1
    run_labels_2 = np.repeat(np.arange(n_trials_2) % 2, T)       # 0/1
    run_labels   = np.concatenate([run_labels_1, run_labels_2])

    ds = Dataset(
        patterns_all,
        obs_descriptors=dict(
            cond = cond_labels,
            run  = run_labels)
    )

    # ------------------------------------------------------------
    # 2.  200 × 200 RDM over 'cond', CV splits by 'run'
    # ------------------------------------------------------------
    full_rdm = calc_rdm_crossnobis(
        ds,
        descriptor    = 'cond',   # single key!
        cv_descriptor = 'run'
    )[0].get_matrices()[0]           # ndarray

    return (full_rdm[0:T, T:2*T],  # states1 - states1 (between)
        full_rdm[0:T, 0:T],        # states2 - states2 (within)
        full_rdm[T:2*T, T:2*T])    # states1 - states2 (within)


def compute_sauerbrei_least_squares_concat(
        firing_rates_simple, 
        input_firing_rates_simple, 
        trial_lengths, 
        times, 
        n_pcs_X,
        n_pcs_y, 
        data_mode, 
        diff_mode):

    ## Concatenate firing rates in trials, then reduce to the first n_pcs_y principal components for the dependent variable
    pca_y = PCA(n_components=n_pcs_y, svd_solver='full')
    firing_rates_simple_pca_y_concat = pca_y.fit_transform(np.concatenate(firing_rates_simple, axis=0))

    ## Revert to the original trial-wise format
    split_indices = np.cumsum(trial_lengths)[:-1]
    firing_rates_simple_pca_y = np.split(firing_rates_simple_pca_y_concat, split_indices, axis=0)

    ## Find the derivative of firing_rates_simple_pca_y with numerical differentiation
    n_trials = len(firing_rates_simple)
    assert n_trials == len(input_firing_rates_simple)
    assert n_trials == len(times)
    
    firing_rates_pca_derivatives = []

    for i_trial in range(n_trials):

        firing_rates = firing_rates_simple_pca_y[i_trial]
        times_ = times[i_trial]

        firing_rates_diff = np.diff(firing_rates, axis=0)
        times_diff = np.diff(times_)
        firing_rates_derivative = firing_rates_diff / times_diff[:, None]

        firing_rates_pca_derivatives.append(firing_rates_derivative)

    firing_rates_pca_derivatives_concat = np.concatenate(firing_rates_pca_derivatives, axis=0)

    ## Concatenate firing rates (while excluding the last or first time point)
    firing_rates_concat       = []
    input_firing_rates_concat = []

    for i_trial in range(n_trials):

        if diff_mode == 'forward':
            firing_rates_concat.append(firing_rates_simple[i_trial][:-1, :])
            input_firing_rates_concat.append(input_firing_rates_simple[i_trial][:-1, :])
        elif diff_mode == 'backward':
            firing_rates_concat.append(firing_rates_simple[i_trial][1:, :])
            input_firing_rates_concat.append(input_firing_rates_simple[i_trial][1:, :])
        else:
            raise ValueError('Invalid diff mode: ' + diff_mode)

    firing_rates_concat       = np.concatenate(firing_rates_concat, axis=0)
    input_firing_rates_concat = np.concatenate(input_firing_rates_concat, axis=0)

    ## Reduce to the first n_pcs principal components for the independent variables
    pca_X1 = PCA(n_components=n_pcs_X, svd_solver='full')
    firing_rates_pca_concat = pca_X1.fit_transform(firing_rates_concat)

    pca_X2 = PCA(n_components=n_pcs_X, svd_solver='full')
    input_firing_rates_pca_concat = pca_X2.fit_transform(input_firing_rates_concat)

    pca_X3 = PCA(n_components=n_pcs_X, svd_solver='full')
    mixed_firing_rates_pca_concat = pca_X3.fit_transform(np.column_stack((firing_rates_concat, input_firing_rates_concat)))

    # n_neurons       = firing_rates_concat.shape[1]
    # n_input_neurons = input_firing_rates_concat.shape[1]

    ## Concatenatethe firing rates and input firing rates
    if data_mode == 'MC':
        X = firing_rates_pca_concat
    elif data_mode == 'PPC':
        X = input_firing_rates_pca_concat
    elif data_mode == 'MC|PPC':
        # X = np.column_stack((firing_rates_pca_concat[:, :n_pcs_X // 2], input_firing_rates_pca_concat[:, :n_pcs_X // 2]))
        X = np.column_stack((firing_rates_pca_concat, input_firing_rates_pca_concat))
    elif data_mode == 'MC&PPC':
        X = mixed_firing_rates_pca_concat
    else:
        raise ValueError('Invalid data mode: ' + data_mode)

    ## Fit least squares
    y = firing_rates_pca_derivatives_concat

    # B = np.linalg.inv(X.T @ X) @ X.T @ y
    B = np.linalg.pinv(X) @ y

    ## Compute R2 
    y_pred = X @ B
    SSR = np.sum((y_pred - y)**2)
    SST = np.sum((y - np.mean(y, axis=0))**2)
    R2 = 1 - SSR / SST

    return R2


def compute_sauerbrei_least_squares_single_trial_avg(
        firing_rates_simple, 
        input_firing_rates_simple, 
        trial_lengths,
        times, 
        n_pcs_X,
        n_pcs_y, 
        data_mode, 
        diff_mode):
    
    ## Concatenate firing rates in trials, then reduce to the first n_pcs principal components
    firing_rates_simple_concat = np.concatenate(firing_rates_simple, axis=0)
    input_firing_rates_simple_concat = np.concatenate(input_firing_rates_simple, axis=0)
    mixed_firing_rates_simple_concat = np.column_stack((firing_rates_simple_concat, input_firing_rates_simple_concat))

    pca_y = PCA(n_components=n_pcs_y, svd_solver='full')
    firing_rates_simple_pca_y_concat = pca_y.fit_transform(firing_rates_simple_concat)

    pca_X1 = PCA(n_components=n_pcs_X, svd_solver='full')
    firing_rates_simple_pca_concat = pca_X1.fit_transform(firing_rates_simple_concat)

    pca_X2 = PCA(n_components=n_pcs_X, svd_solver='full')
    input_firing_rates_simple_pca_concat = pca_X2.fit_transform(input_firing_rates_simple_concat)

    pca_X3 = PCA(n_components=n_pcs_X, svd_solver='full')
    mixed_firing_rates_simple_pca_concat = pca_X3.fit_transform(mixed_firing_rates_simple_concat)

    ## Revert to the original trial-wise format
    split_indices = np.cumsum(trial_lengths)[:-1]
    firing_rates_simple_pca_y     = np.split(firing_rates_simple_pca_y_concat,     split_indices, axis=0)
    firing_rates_simple_pca       = np.split(firing_rates_simple_pca_concat,       split_indices, axis=0)
    input_firing_rates_simple_pca = np.split(input_firing_rates_simple_pca_concat, split_indices, axis=0)
    mixed_firing_rates_simple_pca = np.split(mixed_firing_rates_simple_pca_concat, split_indices, axis=0)

    ## Shape checks
    n_trials = len(firing_rates_simple)
    assert n_trials == len(input_firing_rates_simple)
    assert n_trials == len(times)

    ## Find per trial R2 values
    R2_all = []
    for i_trial in range(n_trials):

        firing_rates_pca_y     = firing_rates_simple_pca_y[i_trial]
        firing_rates_pca       = firing_rates_simple_pca[i_trial]
        input_firing_rates_pca = input_firing_rates_simple_pca[i_trial]
        mixed_firing_rates_pca = mixed_firing_rates_simple_pca[i_trial]
        times_ = times[i_trial]

        ## Find the derivative of firing_rates_pca with numerical differentiation
        firing_rates_diff = np.diff(firing_rates_pca_y, axis=0)
        times_diff = np.diff(times_)
        firing_rates_pca_derivative = firing_rates_diff / times_diff[:, None]

        ## Excluding the last or first time point
        if diff_mode == 'forward':
            firing_rates_pca = firing_rates_pca[:-1, :]
            input_firing_rates_pca = input_firing_rates_pca[:-1, :]
            mixed_firing_rates_pca = mixed_firing_rates_pca[:-1, :]
        elif diff_mode == 'backward':
            firing_rates_pca = firing_rates_pca[1:, :]
            input_firing_rates_pca = input_firing_rates_pca[1:, :]
            mixed_firing_rates_pca = mixed_firing_rates_pca[1:, :]
        else:
            raise ValueError('Invalid diff mode: ' + diff_mode)

        # n_neurons       = firing_rates_concat.shape[1]
        # n_input_neurons = input_firing_rates_concat.shape[1]

        ## Concatenatethe firing rates and input firing rates
        if data_mode == 'MC':
            X = firing_rates_pca
        elif data_mode == 'PPC':
            X = input_firing_rates_pca
        elif data_mode == 'MC|PPC':
            # X = np.column_stack((firing_rates_pca[:, :n_pcs_X // 2], input_firing_rates_pca[:, :n_pcs_X // 2]))
            X = np.column_stack((firing_rates_pca, input_firing_rates_pca))
        elif data_mode == 'MC&PPC':
            X = mixed_firing_rates_pca
        else:
            raise ValueError('Invalid data mode: ' + data_mode)

        ## Fit least squares
        y = firing_rates_pca_derivative

        # B = np.linalg.inv(X.T @ X) @ X.T @ y
        B = np.linalg.pinv(X) @ y

        ## Compute R2 
        y_pred = X @ B
        SSR = np.sum((y_pred - y)**2)
        SST = np.sum((y - np.mean(y, axis=0))**2)
        R2 = 1 - SSR / SST

        R2_all.append(R2)

    return np.mean(R2_all)


def compute_sauerbrei_least_squares_single_trial_avg2(
        firing_rates_simple, 
        input_firing_rates_simple, 
        trial_lengths,
        times, 
        n_pcs_X,
        n_pcs_y, 
        data_mode, 
        diff_mode):

    ## Find per trial R2 values
    R2_all = []
    for i_trial in range(len(firing_rates_simple)):

        firing_rates = firing_rates_simple[i_trial]
        input_firing_rates = input_firing_rates_simple[i_trial]
        mixed_firing_rates = np.column_stack((firing_rates, input_firing_rates))

        pca_y = PCA(n_components=n_pcs_y, svd_solver='full')
        firing_rates_pca_y = pca_y.fit_transform(firing_rates)

        pca_X1 = PCA(n_components=n_pcs_X, svd_solver='full')
        firing_rates_pca = pca_X1.fit_transform(firing_rates)

        pca_X2 = PCA(n_components=n_pcs_X, svd_solver='full')
        input_firing_rates_pca = pca_X2.fit_transform(input_firing_rates)

        pca_X3 = PCA(n_components=n_pcs_X, svd_solver='full')
        mixed_firing_rates_pca = pca_X3.fit_transform(mixed_firing_rates)

        times_ = times[i_trial]

        ## Find the derivative of firing_rates_pca with numerical differentiation
        firing_rates_diff = np.diff(firing_rates_pca_y, axis=0)
        times_diff = np.diff(times_)
        firing_rates_pca_derivative = firing_rates_diff / times_diff[:, None]

        ## Excluding the last or first time point
        if diff_mode == 'forward':
            firing_rates_pca = firing_rates_pca[:-1, :]
            input_firing_rates_pca = input_firing_rates_pca[:-1, :]
            mixed_firing_rates_pca = mixed_firing_rates_pca[:-1, :]
        elif diff_mode == 'backward':
            firing_rates_pca = firing_rates_pca[1:, :]
            input_firing_rates_pca = input_firing_rates_pca[1:, :]
            mixed_firing_rates_pca = mixed_firing_rates_pca[1:, :]
        else:
            raise ValueError('Invalid diff mode: ' + diff_mode)

        # n_neurons       = firing_rates_concat.shape[1]
        # n_input_neurons = input_firing_rates_concat.shape[1]

        ## Concatenatethe firing rates and input firing rates
        if data_mode == 'MC':
            X = firing_rates_pca
        elif data_mode == 'PPC':
            X = input_firing_rates_pca
        elif data_mode == 'MC|PPC':
            # X = np.column_stack((firing_rates_pca[:, :n_pcs_X // 2], input_firing_rates_pca[:, :n_pcs_X // 2]))
            X = np.column_stack((firing_rates_pca, input_firing_rates_pca))
        elif data_mode == 'MC&PPC':
            X = mixed_firing_rates_pca
        else:
            raise ValueError('Invalid data mode: ' + data_mode)

        ## Fit least squares
        y = firing_rates_pca_derivative

        # B = np.linalg.inv(X.T @ X) @ X.T @ y
        B = np.linalg.pinv(X) @ y

        ## Compute R2 
        y_pred = X @ B
        SSR = np.sum((y_pred - y)**2)
        SST = np.sum((y - np.mean(y, axis=0))**2)
        R2 = 1 - SSR / SST

        R2_all.append(R2)

    return np.mean(R2_all)


def compute_sauerbrei_least_squares_avg_trials(
        firing_rates_simple, 
        input_firing_rates_simple, 
        times, 
        n_pcs_X,
        n_pcs_y, 
        data_mode, 
        diff_mode):

    ## Concatenate firing rates in trials, then reduce to the first n_pcs_y principal components for the dependent variable
    n_trials, n_times, n_neurons = firing_rates_simple.shape
    _, _, n_input_neurons = input_firing_rates_simple.shape

    pca_y = PCA(n_components=n_pcs_y, svd_solver='full')
    firing_rates_simple_pca_y_concat = pca_y.fit_transform(firing_rates_simple.reshape(-1, n_neurons))

    ## Revert to the original trial-wise format
    firing_rates_simple_pca_y = firing_rates_simple_pca_y_concat.reshape(n_trials, n_times, n_pcs_y)

    ## Find the derivative of firing_rates_simple_pca_y with numerical differentiation
    firing_rates_pca_derivatives = np.diff(firing_rates_simple_pca_y, axis=1) / np.diff(times[:, :, None], axis=1)

    ## Average firing rates (while excluding the last or first time point)
    if diff_mode == 'forward':
        firing_rates_avg = np.mean(firing_rates_simple[:, :-1, :], axis=0)
        input_firing_rates_avg = np.mean(input_firing_rates_simple[:, :-1, :], axis=0)
    elif diff_mode == 'backward':
        firing_rates_avg = np.mean(firing_rates_simple[:, 1:, :], axis=0)
        input_firing_rates_avg = np.mean(input_firing_rates_simple[:, 1:, :], axis=0)
    else:
        raise ValueError('Invalid diff mode: ' + diff_mode)

    ## Reduce to the first n_pcs principal components for the independent variables
    pca_X1 = PCA(n_components=n_pcs_X, svd_solver='full')
    firing_rates_pca_avg = pca_X1.fit_transform(firing_rates_avg)

    pca_X2 = PCA(n_components=n_pcs_X, svd_solver='full')
    input_firing_rates_pca_avg = pca_X2.fit_transform(input_firing_rates_avg)

    pca_X3 = PCA(n_components=n_pcs_X, svd_solver='full')
    mixed_firing_rates_pca_avg = pca_X3.fit_transform(np.column_stack((firing_rates_avg, input_firing_rates_avg)))

    ## Concatenatethe firing rates and input firing rates
    if data_mode == 'MC':
        X = firing_rates_pca_avg
    elif data_mode == 'PPC':
        X = input_firing_rates_pca_avg
    elif data_mode == 'MC|PPC':
        # X = np.column_stack((firing_rates_pca_avg[:, :n_pcs_X // 2], input_firing_rates_pca_avg[:, :n_pcs_X // 2]))
        X = np.column_stack((firing_rates_pca_avg, input_firing_rates_pca_avg))
    elif data_mode == 'MC&PPC':
        X = mixed_firing_rates_pca_avg
    else:
        raise ValueError('Invalid data mode: ' + data_mode)
    
    ## Average trials
    # X = np.mean(X.reshape(n_trials, n_times - 1, -1), axis=0)
    y = np.mean(firing_rates_pca_derivatives, axis=0)

    ## Fit least squares
    # B = np.linalg.inv(X.T @ X) @ X.T @ y
    B = np.linalg.pinv(X) @ y

    ## Compute R2 
    y_pred = X @ B
    SSR = np.sum((y_pred - y)**2)
    SST = np.sum((y - np.mean(y))**2)
    R2 = 1 - SSR / SST

    return R2


class DataLoader:

    def __init__(
        self, 
        data_dir, 
        results_dir,
        session_data_name, 
        unit_filter, 
        input_unit_filter,
        window_config,
        trial_filter):
        
        self.data_dir          = data_dir
        self.session_data_name = session_data_name
        self.unit_filter       = unit_filter
        self.input_unit_filter = input_unit_filter
        self.window_config     = window_config
        self.trial_filter      = trial_filter

        self.data_path_prefix    = os.path.join(data_dir,    session_data_name)
        self.session_results_dir = os.path.join(results_dir, session_data_name)

        self.trial_cutoff_times = None

    
    def load_firing_rate_data(self):

        firing_rates_file_name = '_'.join(map(str, [x for x in [
            self.data_path_prefix, 
            'aligned_firing_rates', 
            self.unit_filter, 
            self.window_config,
            self.trial_filter] if x is not None]))
        
        with open(firing_rates_file_name + '.pkl', 'rb') as f:
            firing_rates = pickle.load(f)
        
        self.firing_rates = firing_rates[0]

        ## Load input firing rates if applicable
        if self.input_unit_filter is not None:
            firing_rates_file_name = '_'.join(map(str, [x for x in [
                self.data_path_prefix, 
                'aligned_firing_rates', 
                self.input_unit_filter, 
                self.window_config,
                self.trial_filter] if x is not None]))

            with open(firing_rates_file_name + '.pkl', 'rb') as f:
                firing_rates = pickle.load(f)

            self.input_firing_rates = firing_rates[0]
        else:
            self.input_firing_rates = None

        ## Extract trial ids
        self.trial_ids = []
        self.times     = []

        for i in range(len(self.firing_rates)):
            trial_id = self.firing_rates[i].trial_id
            self.trial_ids.append(trial_id)

            self.times.append(np.array(self.firing_rates[i].time))

        self.trial_ids = np.array(self.trial_ids)


    def load_cursor_data(self):

        cursor_data_name = '_'.join(map(str, [x for x in [
            self.data_path_prefix,
            'aligned_cursor',
            self.window_config] if x is not None]))
        
        with open(cursor_data_name + '.pkl', 'rb') as f:
            cursor_data = pickle.load(f)

        cursor_pos, cursor_vel = cursor_data

        self.cursor_pos = cursor_pos[self.trial_ids]
        self.cursor_vel = cursor_vel[self.trial_ids]


    def remove_target_overlap(self, target_radius):
        ## NOTE: This applies to cursor data and firing rates
        ## Remove target-overlap samples, but keep the first “hit”

        target_positions = self.get_target_positions()

        for i, trial_id in enumerate(self.trial_ids):
            pos = self.cursor_pos[trial_id]        # xarray (time × 2)
            vel = self.cursor_vel[trial_id]
            tgt = target_positions[i]         # (2,)

            # Boolean array: True when the cursor is inside the target
            inside = np.linalg.norm(pos - tgt, axis=1) < target_radius

            # Mask that tells xarray WHICH SAMPLES TO KEEP
            keep = np.ones_like(inside, dtype=bool)

            if inside.any():
                keep[inside] = False                  # drop all overlapping samples
                keep[np.argmax(inside)] = True        # …except the very first one

            # Convert to an xarray mask that shares the “time” coord
            keep_cursor = xr.DataArray(
                keep,
                dims="time",
                coords={"time": pos.time}
            )

            # Apply mask and drop the unwanted rows
            self.cursor_pos[trial_id] = pos.where(keep_cursor, drop=True)
            self.cursor_vel[trial_id] = vel.where(keep_cursor, drop=True)

            # Filter out firing rates after the last cursor time
            last_cursor_time = self.cursor_pos[trial_id].time.max().item()

            keep_fr = xr.DataArray(
                self.firing_rates[i].time <= last_cursor_time,   # True up to & incl. last time
                dims="time",
                coords={"time": self.firing_rates[i].time}
            )

            self.firing_rates[i] = self.firing_rates[i].where(keep_fr, drop=True)
            if self.input_firing_rates is not None:
                self.input_firing_rates[i] = self.input_firing_rates[i].where(keep_fr, drop=True)

            # Update self.times
            self.times[i] = self.firing_rates[i].time.values


    def extract_cursor_states_and_times_without_alignment(self):
        cursor_states = []
        cursor_times  = []

        for trial_id in self.trial_ids:
            cursor_pos_ = self.cursor_pos[trial_id].values
            cursor_vel_ = self.cursor_vel[trial_id].values
            cursor_states_ = np.column_stack((cursor_pos_, cursor_vel_))

            cursor_times_ = self.cursor_pos[trial_id].time.values
            assert np.all(cursor_times_ == self.cursor_vel[trial_id].time.values)
            cursor_times_ = np.array(cursor_times_)

            cursor_states.append(cursor_states_)
            cursor_times.append(cursor_times_)

        return cursor_states, cursor_times

    
    def compute_masked_trial_cutoff_times(self, time_limit, distance_limit):
        
        ## Load cursor positions
        cursor_data_name = '_'.join(map(str, [x for x in [
            self.data_path_prefix,
            'aligned_cursor',
            self.window_config] if x is not None]))

        with open(cursor_data_name + '.pkl', 'rb') as f:
            cursor_data = pickle.load(f)

        cursor_pos, _ = cursor_data
        cursor_pos = cursor_pos[self.trial_ids]

        ## Load target positions
        trials_df = pd.read_csv(self.data_path_prefix + '_trials.csv')

        target_positions_x = trials_df['target_pos_x']
        target_positions_y = trials_df['target_pos_y']
        target_positions   = np.column_stack((target_positions_x, target_positions_y))
        target_positions   = target_positions[self.trial_ids]

        self.trial_cutoff_times = []

        for i, trial_id in enumerate(self.trial_ids):
            
            ## Compute the distance between the cursor and the target
            cursor_pos_ = cursor_pos[trial_id]
            target_pos_ = target_positions[i]
            cursor_target_distance = np.sqrt(np.sum((cursor_pos_ - target_pos_)**2, axis=1))

            ## Find the time before time limit where the cursor is closest to the target
            cursor_pos_time_ = np.array(cursor_pos_.time)
            time_limit_idx = np.argmax(cursor_pos_time_[cursor_pos_time_ < time_limit])

            ## Find the minimum distance before time limit
            cursor_target_distance_ = cursor_target_distance[:time_limit_idx]
            min_distance = np.min(cursor_target_distance_)

            if min_distance < distance_limit:
                ## Record the time where the cursor is closest to the target
                min_distance_idx = np.argmin(np.array(cursor_target_distance_))
                self.trial_cutoff_times.append(cursor_pos_time_[min_distance_idx])
            else:
                self.trial_cutoff_times.append(None)
    

    def reformat_firing_rate_data(
        self, 
        data_format, 
        index_buffer=0, 
        time_offset=0, 
        zero_padding=False, 
        time_compensation=0, 
        trial_length_filter_percentile=None):

        ## Extract firing rates
        self.firing_rates, firing_rates_simple, self.trial_ids = extract_trial_firing_rates(
            self.firing_rates, 
            trial_number_filter=None, 
            cutoff_times=self.trial_cutoff_times)

        if data_format == 'goresample':
            index_buffer = 25

        ## Filter out trials with the smallest trial lengths
        if trial_length_filter_percentile is not None:
            firing_rates_simple, trial_lengths, trial_keep_mask = filter_by_trial_length_percentile(firing_rates_simple, percentile=trial_length_filter_percentile)
            self.firing_rates = [self.firing_rates[i] for i in range(len(self.firing_rates)) if trial_keep_mask[i]]
        else:
            n_trials = len(firing_rates_simple)
            trial_keep_mask = np.ones(n_trials, dtype=bool)

        ## Adjust data format
        self.times_new = None

        if data_format is None:
            trial_lengths = np.array([firing_rates.shape[0] for firing_rates in firing_rates_simple])
            self.times_new = self.times
        elif data_format == 'fill0':
            firing_rates_simple, trial_lengths = fill_emissions(firing_rates_simple)
        elif data_format == 'truncate_front':
            firing_rates_simple, trial_lengths, self.times_new = truncate_emissions(self.firing_rates, truncate_end=False)
        elif data_format == 'truncate_end':
            firing_rates_simple, trial_lengths, self.times_new = truncate_emissions(self.firing_rates, truncate_end=True)
        elif data_format in ['resample', 'resample_avg', 'goresample']:
            firing_rates_simple, trial_lengths, self.times_new = resample_emissions(self.firing_rates, trial_length_end_buffer=index_buffer)
        else:
            raise ValueError('Invalid data format: ' + data_format)
        
        if self.input_firing_rates is not None:
            self.input_firing_rates, input_firing_rates_simple, _ = extract_trial_firing_rates(
                self.input_firing_rates, 
                trial_number_filter=None, 
                cutoff_times=self.trial_cutoff_times)
            
            if trial_length_filter_percentile is not None:
                input_firing_rates_simple = [input_firing_rates_simple[i] for i in range(len(input_firing_rates_simple)) if trial_keep_mask[i]]
                self.input_firing_rates = [self.input_firing_rates[i] for i in range(len(self.input_firing_rates)) if trial_keep_mask[i]]

            if data_format is None:
                pass
            elif data_format == 'fill0':
                input_firing_rates_simple, _ = fill_emissions(input_firing_rates_simple)
            elif data_format == 'truncate_front':
                input_firing_rates_simple, _, _ = truncate_emissions(self.input_firing_rates, truncate_end=False)
            elif data_format == 'truncate_end':
                input_firing_rates_simple, _, _ = truncate_emissions(self.input_firing_rates, truncate_end=True)
            elif data_format in ['resample', 'resample_avg', 'goresample']:
                input_firing_rates_simple, _, _ = resample_emissions(self.input_firing_rates, trial_length_end_buffer=index_buffer)
            else:
                raise ValueError('Invalid data format: ' + data_format)
            
            ## Shift input firing rates by time_offset relative to firing rates
            ## time_offset > 0 means firing rates are delayed relative to input firing rates (PPC leads MC)
            ## time_offset < 0 means firing rates are advanced relative to input firing rates (MC leads PPC)
            if time_offset > 0 and not zero_padding:
                firing_rates_simple = [firing_rates_simple_[time_offset:] for firing_rates_simple_ in firing_rates_simple]
                input_firing_rates_simple = [input_firing_rates_simple_[:-time_offset] for input_firing_rates_simple_ in input_firing_rates_simple]

                trial_lengths = [trial_length - time_offset for trial_length in trial_lengths]
                self.times_new = [times_[time_offset:] for times_ in self.times_new]

            elif time_offset < 0 and not zero_padding:
                firing_rates_simple = [firing_rates_simple_[:time_offset] for firing_rates_simple_ in firing_rates_simple]
                input_firing_rates_simple = [input_firing_rates_simple_[-time_offset:] for input_firing_rates_simple_ in input_firing_rates_simple]

                trial_lengths = [trial_length + time_offset for trial_length in trial_lengths]
                self.times_new = [times_[:time_offset] for times_ in self.times_new]

            elif time_offset > 0 and zero_padding:
                n_input_neurons = input_firing_rates_simple[0].shape[1]
                padding = np.zeros((abs(time_offset), n_input_neurons))
                input_firing_rates_simple = [np.concatenate((padding, input_firing_rates_simple_[:-time_offset]), axis=0) for input_firing_rates_simple_ in input_firing_rates_simple]

            elif time_offset < 0 and zero_padding: 
                n_input_neurons = input_firing_rates_simple[0].shape[1]
                padding = np.zeros((abs(time_offset), n_input_neurons))
                input_firing_rates_simple = [np.concatenate((input_firing_rates_simple_[-time_offset:], padding), axis=0) for input_firing_rates_simple_ in input_firing_rates_simple]

            if time_compensation > 0:

                firing_rates_simple = [firing_rates_simple_[:-time_compensation] for firing_rates_simple_ in firing_rates_simple]
                input_firing_rates_simple = [input_firing_rates_simple_[:-time_compensation] for input_firing_rates_simple_ in input_firing_rates_simple]

                trial_lengths = [trial_length - time_compensation for trial_length in trial_lengths]
                self.times_new = [times_[:-time_compensation] for times_ in self.times_new]

        else:
            input_firing_rates_simple = None

        
        self.trial_ids = self.trial_ids[trial_keep_mask]

        return (
            firing_rates_simple, 
            input_firing_rates_simple,
            self.trial_ids,
            trial_lengths,
            self.times_new,
            trial_keep_mask)


    def align_cursor_to_firing_rates(self):
        
        ## Align cursor times to firing rate times
        self.cursor_states = align_cursor_to_fr_times(
            self.firing_rates, 
            self.cursor_pos, 
            self.cursor_vel, 
            self.trial_ids,
            self.times_new)

        return self.cursor_states
    

    def resample_cursor_data(self, trial_length_new):

        n_trials = len(self.cursor_pos)
        assert n_trials == len(self.cursor_vel)

        cursor_states = np.empty((n_trials, trial_length_new, 4), dtype=float)

        for i, (cp, cv) in enumerate(zip(self.cursor_pos, self.cursor_vel)):
            # Convert xarray → NumPy (T × 2)
            pos = cp.values
            vel = cv.values

            # Pure-NumPy linear interpolation --------------------------
            t_old = np.linspace(0.0, 1.0, pos.shape[0],     endpoint=False)
            t_new = np.linspace(0.0, 1.0, trial_length_new, endpoint=False)

            pos_rs = np.column_stack([
                np.interp(t_new, t_old, pos[:, 0]),   # x
                np.interp(t_new, t_old, pos[:, 1])    # y
            ])

            vel_rs = np.column_stack([
                np.interp(t_new, t_old, vel[:, 0]),   # vx
                np.interp(t_new, t_old, vel[:, 1])    # vy
            ])

            # Pack into (x, y, vx, vy)
            cursor_states[i, :, :2] = pos_rs
            cursor_states[i, :, 2:] = vel_rs

        return cursor_states
    

    def get_target_ids(self):

        trials_df = pd.read_csv(self.data_path_prefix + '_trials.csv')
        
        target_ids = trials_df['target_index'].to_numpy()
        target_ids = target_ids[self.trial_ids]

        return target_ids
    

    def get_target_positions(self):
            
        trials_df = pd.read_csv(self.data_path_prefix + '_trials.csv')
    
        target_positions_x = trials_df['target_pos_x']
        target_positions_y = trials_df['target_pos_y']
        target_positions = np.column_stack((target_positions_x, target_positions_y))
        target_positions = target_positions[self.trial_ids]
    
        return target_positions
    

    def compute_target_overlapping_time_filters(self, target_radius=0.1):
        n_trials = len(self.cursor_states)

        target_positions = self.get_target_positions()

        overlapping_time_filters = []

        for i_trial in range(n_trials):
            cursor_positions = self.cursor_states[i_trial][:, 0:2]
            target_position = target_positions[i_trial]
            distance_to_target = np.sqrt(np.sum((cursor_positions - target_position)**2, axis=1))

            overlapping_time_filter = (distance_to_target > target_radius)
            overlapping_time_filters.append(overlapping_time_filter)

        return overlapping_time_filters


    def reformat_cursor_data(self, label_format, data_format):

        ## Load target positions
        target_positions = self.get_target_positions()

        ## Reformat cursor states according to label format
        if label_format == 'cartesian':
            cursor_states  = self.cursor_states
            n_labels       = 4
            n_labels_final = 6
        
        elif label_format == 'polar':
            cursor_states  = convert_states_to_polar(self.cursor_states)
            n_labels       = 4
            n_labels_final = 4
        
        elif label_format == 'uvd2tc':
            cursor_states  = compute_univec_direction_to_target(self.cursor_states, target_positions, polar=False)
            n_labels       = 2
            n_labels_final = 2
        
        elif label_format == 'uvd2tp':
            cursor_states  = compute_univec_direction_to_target(self.cursor_states, target_positions, polar=True)
            n_labels       = 2
            n_labels_final = 2

        else:
            raise ValueError('Invalid label format: ', label_format)

        if data_format in ['truncate_front', 'truncate_end', 'resample']:
            cursor_states = np.array(cursor_states)

        return cursor_states, n_labels, n_labels_final


    def get_model_result_dir(
        self,
        time_offset=None,
        data_format=None,
        train_test=None,
        model_type=None,
        kernel_type=None,
        dynamics_class=None,
        emission_class=None,
        likelihood_class=None,
        method_type=None,
        init_type=None,
        subspace_type=None,
        alpha=None,
        special_flag=None,
        check_existence=True):

        model_dir_name = '_'.join(map(str, [x for x in [
            self.unit_filter,
            self.input_unit_filter,
            self.window_config,
            time_offset,
            data_format,
            self.trial_filter,
            train_test,
            model_type,
            kernel_type,
            dynamics_class,
            emission_class,
            likelihood_class,
            method_type,
            init_type,
            subspace_type,
            alpha,
            special_flag] if x is not None]))

        model_results_dir = os.path.join(self.session_results_dir, model_dir_name)
        
        if not os.path.isdir(model_results_dir):
            if check_existence:
                raise ValueError('Model results directory does not exist: ', model_results_dir)
            else:
                os.makedirs(model_results_dir)
        
        return model_results_dir



class DataLoaderDuo:

    def __init__(
        self, 
        data_dir, 
        results_dir,
        session_data_name, 
        unit_filter, 
        input_unit_filter,
        window_config,
        trial_filters):
        
        self.data_dir          = data_dir
        self.session_data_name = session_data_name
        self.unit_filter       = unit_filter
        self.input_unit_filter = input_unit_filter
        self.window_config     = window_config
        self.trial_filters     = trial_filters

        self.data_path_prefix    = os.path.join(data_dir,    session_data_name)
        self.session_results_dir = os.path.join(results_dir, session_data_name)

    
    def load_firing_rate_data(self):

        firing_rates_file_name_1 = '_'.join(map(str, [x for x in [
            self.data_path_prefix, 
            'aligned_firing_rates', 
            self.unit_filter, 
            self.window_config,
            self.trial_filters[0]] if x is not None]))

        firing_rates_file_name_2 = '_'.join(map(str, [x for x in [
            self.data_path_prefix, 
            'aligned_firing_rates', 
            self.unit_filter, 
            self.window_config,
            self.trial_filters[1]] if x is not None]))
        
        with open(firing_rates_file_name_1 + '.pkl', 'rb') as f:
            firing_rates_1 = pickle.load(f)

        with open(firing_rates_file_name_2 + '.pkl', 'rb') as f:
            firing_rates_2 = pickle.load(f)
        
        self.firing_rates_1 = firing_rates_1[0]
        self.firing_rates_2 = firing_rates_2[0]

        ## Load input firing rates if applicable
        if self.input_unit_filter is not None:
            input_firing_rates_file_name_1 = '_'.join(map(str, [x for x in [
                self.data_path_prefix, 
                'aligned_firing_rates', 
                self.input_unit_filter, 
                self.window_config,
                self.trial_filters[0]] if x is not None]))

            input_firing_rates_file_name_2 = '_'.join(map(str, [x for x in [
                self.data_path_prefix, 
                'aligned_firing_rates', 
                self.input_unit_filter, 
                self.window_config,
                self.trial_filters[1]] if x is not None]))

            with open(input_firing_rates_file_name_1 + '.pkl', 'rb') as f:
                firing_rates_1 = pickle.load(f)

            with open(input_firing_rates_file_name_2 + '.pkl', 'rb') as f:
                firing_rates_2 = pickle.load(f)

            self.input_firing_rates_1 = firing_rates_1[0]
            self.input_firing_rates_2 = firing_rates_2[0]
        else:
            self.input_firing_rates_1 = None
            self.input_firing_rates_2 = None

        ## Extract trial attributes
        self.trial_ids_1 = []
        self.trial_ids_2 = []
        self.times_1     = []
        self.times_2     = []

        for i in range(len(self.firing_rates_1)):
            trial_id = self.firing_rates_1[i].trial_id
            self.trial_ids_1.append(trial_id)
            self.times_1.append(np.array(self.firing_rates_1[i].time))

        for i in range(len(self.firing_rates_2)):
            trial_id = self.firing_rates_2[i].trial_id
            self.trial_ids_2.append(trial_id)
            self.times_2.append(np.array(self.firing_rates_2[i].time))

        ## Filter out the trials that are not selected
        self.trial_ids_1 = np.array(self.trial_ids_1)
        self.trial_ids_2 = np.array(self.trial_ids_2)

    
    def load_cursor_data(self):

        cursor_data_name = '_'.join(map(str, [x for x in [
            self.data_path_prefix,
            'aligned_cursor',
            self.window_config] if x is not None]))

        with open(cursor_data_name + '.pkl', 'rb') as f:
            cursor_data = pickle.load(f)

        cursor_pos, cursor_vel = cursor_data

        self.cursor_pos_1 = cursor_pos[self.trial_ids_1]
        self.cursor_pos_2 = cursor_pos[self.trial_ids_2]
        self.cursor_vel_1 = cursor_vel[self.trial_ids_1]
        self.cursor_vel_2 = cursor_vel[self.trial_ids_2]

    
    def remove_target_overlap(self, target_radius):
        ## NOTE: This applies to cursor data and firing rates
        ## Remove target-overlap samples, but keep the first “hit”

        target_positions_1, target_positions_2 = self.get_target_positions()

        for i, trial_id in enumerate(self.trial_ids_1):
            pos = self.cursor_pos_1[trial_id]        # xarray (time × 2)
            vel = self.cursor_vel_1[trial_id]
            tgt = target_positions_1[i]         # (2,)

            # Boolean array: True when the cursor is inside the target
            inside = np.linalg.norm(pos - tgt, axis=1) < target_radius

            # Mask that tells xarray WHICH SAMPLES TO KEEP
            keep = np.ones_like(inside, dtype=bool)

            if inside.any():
                keep[inside] = False                  # drop all overlapping samples
                keep[np.argmax(inside)] = True        # …except the very first one

            # Convert to an xarray mask that shares the “time” coord
            keep_cursor = xr.DataArray(
                keep,
                dims="time",
                coords={"time": pos.time}
            )

            # Apply mask and drop the unwanted rows
            self.cursor_pos_1[trial_id] = pos.where(keep_cursor, drop=True)
            self.cursor_vel_1[trial_id] = vel.where(keep_cursor, drop=True)

            # Filter out firing rates after the last cursor time
            last_cursor_time = self.cursor_pos_1[trial_id].time.max().item()

            keep_fr = xr.DataArray(
                self.firing_rates_1[i].time <= last_cursor_time,   # True up to & incl. last time
                dims="time",
                coords={"time": self.firing_rates_1[i].time}
            )

            self.firing_rates_1[i] = self.firing_rates_1[i].where(keep_fr, drop=True)
            if self.input_firing_rates_1 is not None:
                self.input_firing_rates_1[i] = self.input_firing_rates_1[i].where(keep_fr, drop=True)

            # Update self.times
            self.times_1[i] = self.firing_rates_1[i].time.values

        ## Repeat for the second set of trials
        for i, trial_id in enumerate(self.trial_ids_2):
            pos = self.cursor_pos_2[trial_id]        # xarray (time × 2)
            vel = self.cursor_vel_2[trial_id]
            tgt = target_positions_2[i]         # (2,)

            # Boolean array: True when the cursor is inside the target
            inside = np.linalg.norm(pos - tgt, axis=1) < target_radius

            # Mask that tells xarray WHICH SAMPLES TO KEEP
            keep = np.ones_like(inside, dtype=bool)

            if inside.any():
                keep[inside] = False                  # drop all overlapping samples
                keep[np.argmax(inside)] = True        # …except the very first one

            # Convert to an xarray mask that shares the “time” coord
            keep_cursor = xr.DataArray(
                keep,
                dims="time",
                coords={"time": pos.time}
            )

            # Apply mask and drop the unwanted rows
            self.cursor_pos_2[trial_id] = pos.where(keep_cursor, drop=True)
            self.cursor_vel_2[trial_id] = vel.where(keep_cursor, drop=True)

            # Filter out firing rates after the last cursor time
            last_cursor_time = self.cursor_pos_2[trial_id].time.max().item()

            keep_fr = xr.DataArray(
                self.firing_rates_2[i].time <= last_cursor_time,   # True up to & incl. last time
                dims="time",
                coords={"time": self.firing_rates_2[i].time}
            )

            self.firing_rates_2[i] = self.firing_rates_2[i].where(keep_fr, drop=True)
            if self.input_firing_rates_2 is not None:
                self.input_firing_rates_2[i] = self.input_firing_rates_2[i].where(keep_fr, drop=True)

            # Update self.times
            self.times_2[i] = self.firing_rates_2[i].time.values
    

    def extract_cursor_states_and_times_without_alignment(self):
        cursor_states_1 = []
        cursor_states_2 = []
        cursor_times_1  = []
        cursor_times_2  = []

        for trial_id in self.trial_ids_1:
            cursor_pos_ = self.cursor_pos[trial_id].values
            cursor_vel_ = self.cursor_vel[trial_id].values
            cursor_states_ = np.column_stack((cursor_pos_, cursor_vel_))

            cursor_times_ = self.cursor_pos[trial_id].time.values
            assert np.all(cursor_times_ == self.cursor_vel[trial_id].time.values)
            cursor_times_ = np.array(cursor_times_)

            cursor_states_1.append(cursor_states_)
            cursor_times_1.append(cursor_times_)

        for trial_id in self.trial_ids_2:
            cursor_pos_ = self.cursor_pos[trial_id].values
            cursor_vel_ = self.cursor_vel[trial_id].values
            cursor_states_ = np.column_stack((cursor_pos_, cursor_vel_))

            cursor_times_ = self.cursor_pos[trial_id].time.values
            assert np.all(cursor_times_ == self.cursor_vel[trial_id].time.values)
            cursor_times_ = np.array(cursor_times_)

            cursor_states_2.append(cursor_states_)
            cursor_times_2.append(cursor_times_)

        return cursor_states_1, cursor_states_2, cursor_times_1, cursor_times_2


    def reformat_firing_rate_data(
        self, 
        data_format,
        index_buffer=0, 
        trial_length_filter_percentile=None, 
        resample_trial_length=None):

        ## Extract firing rates
        self.firing_rates_1, firing_rates_simple_1, self.trial_ids_1 = extract_trial_firing_rates(
            self.firing_rates_1, 
            trial_number_filter=None, 
            cutoff_times=None)
        
        self.firing_rates_2, firing_rates_simple_2, self.trial_ids_2 = extract_trial_firing_rates(
            self.firing_rates_2,
            trial_number_filter=None,
            cutoff_times=None)
        
        ## Filter out trials with the smallest trial lengths
        if trial_length_filter_percentile is not None:
            firing_rates_simple_1, trial_lengths_1, self.trial_keep_mask_1 = filter_by_trial_length_percentile(firing_rates_simple_1, percentile=trial_length_filter_percentile)
            firing_rates_simple_2, trial_lengths_2, self.trial_keep_mask_2 = filter_by_trial_length_percentile(firing_rates_simple_2, percentile=trial_length_filter_percentile)
        else:
            n_trials_1 = len(firing_rates_simple_1)
            n_trials_2 = len(firing_rates_simple_2)
            self.trial_keep_mask_1 = np.ones(n_trials_1, dtype=bool)
            self.trial_keep_mask_2 = np.ones(n_trials_2, dtype=bool)

        ## Adjust data format
        self.times_new_1 = None
        self.times_new_2 = None

        if data_format is None:
            trial_lengths_1 = np.array([firing_rates.shape[0] for firing_rates in firing_rates_simple_1])
            trial_lengths_2 = np.array([firing_rates.shape[0] for firing_rates in firing_rates_simple_2])
        elif data_format == 'fill0':
            firing_rates_simple_1, trial_lengths_1 = fill_emissions(firing_rates_simple_1)
            firing_rates_simple_2, trial_lengths_2 = fill_emissions(firing_rates_simple_2)
        elif data_format == 'truncate_front':
            firing_rates_simple_1, trial_lengths_1, self.times_new_1, self.trial_keep_mask_1 = truncate_emissions(self.firing_rates_1, truncate_end=False)
            firing_rates_simple_2, trial_lengths_2, self.times_new_2, self.trial_keep_mask_2 = truncate_emissions(self.firing_rates_2, truncate_end=False)
        elif data_format == 'truncate_end':
            firing_rates_simple_1, trial_lengths_1, self.times_new_1, self.trial_keep_mask_1 = truncate_emissions(self.firing_rates_1, truncate_end=True)
            firing_rates_simple_2, trial_lengths_2, self.times_new_2, self.trial_keep_mask_2 = truncate_emissions(self.firing_rates_2, truncate_end=True)
        elif data_format == 'resample':
            firing_rates_simple_1, trial_lengths_1, self.times_new_1 = resample_emissions(self.firing_rates_1, trial_length_new=resample_trial_length, trial_length_end_buffer=index_buffer)
            firing_rates_simple_2, trial_lengths_2, self.times_new_2 = resample_emissions(self.firing_rates_2, trial_length_new=resample_trial_length, trial_length_end_buffer=index_buffer)
        else:
            raise ValueError('Invalid data format: ' + data_format)
        
        if self.input_firing_rates_1 is not None and self.input_firing_rates_2 is not None:
            self.input_firing_rates_1, input_firing_rates_simple_1, _ = extract_trial_firing_rates(
                self.input_firing_rates_1, 
                trial_number_filter=None, 
                cutoff_times=None)
            
            self.input_firing_rates_2, input_firing_rates_simple_2, _ = extract_trial_firing_rates(
                self.input_firing_rates_2,
                trial_number_filter=None,
                cutoff_times=None)
            
            if trial_length_filter_percentile is not None:
                input_firing_rates_simple_1 = [input_firing_rates_simple_1[i] for i in range(len(input_firing_rates_simple_1)) if self.trial_keep_mask_1[i]]
                self.input_firing_rates_1 = [self.input_firing_rates_1[i] for i in range(len(self.input_firing_rates_1)) if self.trial_keep_mask_1[i]]

                input_firing_rates_simple_2 = [input_firing_rates_simple_2[i] for i in range(len(input_firing_rates_simple_2)) if self.trial_keep_mask_2[i]]
                self.input_firing_rates_2 = [self.input_firing_rates_2[i] for i in range(len(self.input_firing_rates_2)) if self.trial_keep_mask_2[i]]

            if data_format is None:
                pass
            elif data_format == 'fill0':
                input_firing_rates_simple_1, _ = fill_emissions(input_firing_rates_simple_1)
                input_firing_rates_simple_2, _ = fill_emissions(input_firing_rates_simple_2)
            elif data_format == 'truncate_front':
                input_firing_rates_simple_1, _, _, _ = truncate_emissions(self.input_firing_rates_1, truncate_end=False)
                input_firing_rates_simple_2, _, _, _ = truncate_emissions(self.input_firing_rates_2, truncate_end=False)
            elif data_format == 'truncate_end':
                input_firing_rates_simple_1, _, _, _ = truncate_emissions(self.input_firing_rates_1, truncate_end=True)
                input_firing_rates_simple_2, _, _, _ = truncate_emissions(self.input_firing_rates_2, truncate_end=True)
            elif data_format == 'resample':
                input_firing_rates_simple_1, _, _ = resample_emissions(self.input_firing_rates_1, trial_length_new=resample_trial_length, trial_length_end_buffer=index_buffer)
                input_firing_rates_simple_2, _, _ = resample_emissions(self.input_firing_rates_2, trial_length_new=resample_trial_length, trial_length_end_buffer=index_buffer)
            else:
                raise ValueError('Invalid data format: ' + data_format)

        else:
            input_firing_rates_simple_1 = None
            input_firing_rates_simple_2 = None

        ## Filter out the trials that are not selected
        self.trial_ids_1 = self.trial_ids_1[self.trial_keep_mask_1]
        self.trial_ids_2 = self.trial_ids_2[self.trial_keep_mask_2]

        self.firing_rates_1 = [self.firing_rates_1[i] for i in range(len(self.firing_rates_1)) if self.trial_keep_mask_1[i]]
        self.firing_rates_2 = [self.firing_rates_2[i] for i in range(len(self.firing_rates_2)) if self.trial_keep_mask_2[i]]

        return (
            firing_rates_simple_1,
            firing_rates_simple_2, 
            input_firing_rates_simple_1,
            input_firing_rates_simple_2,
            self.trial_ids_1,
            self.trial_ids_2,
            trial_lengths_1,
            trial_lengths_2,
            self.times_new_1,
            self.times_new_2,
            self.trial_keep_mask_1,
            self.trial_keep_mask_2)
        

    def align_cursor_to_firing_rates(self):
        self.cursor_pos_1 = self.cursor_pos_1[self.trial_keep_mask_1]
        self.cursor_pos_2 = self.cursor_pos_2[self.trial_keep_mask_2]
        self.cursor_vel_1 = self.cursor_vel_1[self.trial_keep_mask_1]
        self.cursor_vel_2 = self.cursor_vel_2[self.trial_keep_mask_2]

        ## Align cursor times to firing rate times
        self.cursor_states_1 = align_cursor_to_fr_times(
            self.firing_rates_1, 
            self.cursor_pos_1, 
            self.cursor_vel_1, 
            self.trial_ids_1,
            self.times_new_1)

        self.cursor_states_2 = align_cursor_to_fr_times(
            self.firing_rates_2, 
            self.cursor_pos_2, 
            self.cursor_vel_2, 
            self.trial_ids_2, 
            self.times_new_2)
        
        return self.cursor_states_1, self.cursor_states_2
    

    def get_target_ids(self):
        trials_df = pd.read_csv(self.data_path_prefix + '_trials.csv')
        target_ids = trials_df['target_index'].to_numpy()

        target_ids_1 = target_ids[self.trial_ids_1]
        target_ids_2 = target_ids[self.trial_ids_2]

        return target_ids_1, target_ids_2
    

    def get_target_positions(self):
            
        trials_df = pd.read_csv(self.data_path_prefix + '_trials.csv')
    
        target_positions_x = trials_df['target_pos_x']
        target_positions_y = trials_df['target_pos_y']
        target_positions   = np.column_stack((target_positions_x, target_positions_y))

        target_positions_1 = target_positions[self.trial_ids_1]
        target_positions_2 = target_positions[self.trial_ids_2]
    
        return target_positions_1, target_positions_2
    

    def compute_target_overlapping_time_filters(self, target_radius=0.1):
        n_trials_1 = len(self.cursor_states_1)
        n_trials_2 = len(self.cursor_states_2)

        target_positions_1, target_positions_2 = self.get_target_positions()

        overlapping_time_filters_1 = []
        overlapping_time_filters_2 = []

        for i_trial in range(n_trials_1):
            cursor_positions = self.cursor_states_1[i_trial][:, 0:2]
            target_position = target_positions_1[i_trial]
            distance_to_target = np.sqrt(np.sum((cursor_positions - target_position)**2, axis=1))

            overlapping_time_filter = (distance_to_target > target_radius)
            overlapping_time_filters_1.append(overlapping_time_filter)

        for i_trial in range(n_trials_2):
            cursor_positions = self.cursor_states_2[i_trial][:, 0:2]
            target_position = target_positions_2[i_trial]
            distance_to_target = np.sqrt(np.sum((cursor_positions - target_position)**2, axis=1))

            overlapping_time_filter = (distance_to_target > target_radius)
            overlapping_time_filters_2.append(overlapping_time_filter)

        return overlapping_time_filters_1, overlapping_time_filters_2


    def reformat_cursor_data(self, label_format):

        ## Load target positions
        target_positions_1, target_positions_2 = self.get_target_positions()

        ## Reformat cursor states according to label format
        if label_format == 'cartesian':
            cursor_states_1 = self.cursor_states_1
            cursor_states_2 = self.cursor_states_2
            n_labels       = 4
            n_labels_final = 6

        elif label_format in ['pe', 'pe_mag']:
            cursor_states_1 = compute_direction_to_target(self.cursor_states_1, target_positions_1, univec=False)
            cursor_states_2 = compute_direction_to_target(self.cursor_states_2, target_positions_2, univec=False)
            n_labels       = 2
            n_labels_final = 2
        
        elif label_format == 'univec':
            cursor_states_1 = compute_direction_to_target(self.cursor_states_1, target_positions_1, univec=True)
            cursor_states_2 = compute_direction_to_target(self.cursor_states_2, target_positions_2, univec=True)
            n_labels       = 2
            n_labels_final = 2
        
        else:
            raise ValueError('Invalid label format: ', label_format)

        return cursor_states_1, cursor_states_2, n_labels, n_labels_final

    def get_model_result_dirs(
        self,
        time_offset=None,
        data_format=None,
        train_test=None,
        model_type=None,
        kernel_type=None,
        dynamics_class=None,
        emission_class=None,
        likelihood_class=None,
        method_type=None,
        init_type=None,
        subspace_type=None,
        alpha=None,
        check_existence=True):
        
        model_dir_name_1 = '_'.join(map(str, [x for x in [
            self.unit_filter,
            self.input_unit_filter,
            self.window_config,
            # time_offset,
            data_format,
            self.trial_filters[0],
            train_test,
            model_type,
            kernel_type,
            dynamics_class,
            emission_class,
            likelihood_class,
            method_type,
            init_type,
            subspace_type,
            alpha] if x is not None]))
        
        model_dir_name_2 = '_'.join(map(str, [x for x in [
            self.unit_filter,
            self.input_unit_filter,
            self.window_config,
            # time_offset,
            data_format,
            self.trial_filters[1],
            train_test,
            model_type,
            kernel_type,
            dynamics_class,
            emission_class,
            likelihood_class,
            method_type,
            init_type,
            subspace_type,
            alpha] if x is not None]))

        model_results_dir_1 = os.path.join(self.session_results_dir, model_dir_name_1)
        model_results_dir_2 = os.path.join(self.session_results_dir, model_dir_name_2)

        
        if not os.path.isdir(model_results_dir_1):
            if check_existence:
                raise ValueError('Model results directory does not exist: ', model_results_dir_1)
            else:
                os.makedirs(model_results_dir_1)

        if not os.path.isdir(model_results_dir_2):
            if check_existence:
                raise ValueError('Model results directory does not exist: ', model_results_dir_2)
            else:
                os.makedirs(model_results_dir_2)
        
        return model_results_dir_1, model_results_dir_2
