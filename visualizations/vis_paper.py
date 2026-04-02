import os

import pickle
import itertools
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
from scipy.stats import t, wilcoxon, mannwhitneyu, ttest_1samp, ttest_rel
from statsmodels.stats.multitest import multipletests

from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import matplotlib.collections as mcoll
import utils.utils_processing as utils_processing
import utils.utils_vis as utils_vis
import scripts.config as config
from visualizations.vis_config import *



## Read parameters from config
data_dir           = config.data_dir
results_dir        = config.results_dir
vis_dir            = config.vis_dir
session_data_dict  = config.session_data_dict
session_data_names = config.session_data_names

unit_filters       = config.unit_filters
input_unit_filters = config.input_unit_filters
window_config      = config.window_config
time_offset        = config.time_offset
time_step          = config.time_step
data_formats       = config.data_formats
label_formats      = config.label_formats
trial_filters      = config.trial_filters
train_test_options = config.train_test_options

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
subspace_types     = config.subspace_types
alphas             = config.alphas

n_ns_continuous_states = len(ns_states)
n_ns_discrete_states   = len(ns_discrete_states)
n_ns_iters             = len(ns_iters)


# =====================================================================
# Crossnobis cache helpers
# =====================================================================

def _within_crossnobis_cache_path(
        session_data_name, unit_filter, window_config, trial_filter,
        per_target=False, suffix=''):
    """Build path for cached within-condition crossnobis matrices.

    Pattern: {results_dir}/{session}/crossnobis_matrices/
             {unit_filter}/{window_config}/{trial_filter}[_per_target][suffix].pkl
    """
    cache_dir = os.path.join(
        results_dir, session_data_name, 'crossnobis_matrices',
        unit_filter, window_config)
    pt = '_per_target' if per_target else ''
    return os.path.join(cache_dir, f'{trial_filter}{pt}{suffix}.pkl')


def _cross_condition_crossnobis_cache_path(
        session_data_name, unit_filter, window_config, trial_filters,
        per_target=False):
    """Build path for cached cross-condition crossnobis matrices."""
    cache_dir = os.path.join(
        results_dir, session_data_name, 'crossnobis_matrices',
        unit_filter, window_config)
    pt = '_per_target' if per_target else ''
    return os.path.join(
        cache_dir, f'cross_{trial_filters[0]}_{trial_filters[1]}{pt}.pkl')


def _load_or_compute_within_crossnobis(
        session_data_name, unit_filter, window_config, trial_filter,
        data_format, time_step, truncate_percentile,
        pre_start_idx, post_reach_idx,
        per_target=False, per_target_stack=False,
        recompute=False, cache_suffix='',
        trim_buffers=True):
    """Load cached or compute within-condition crossnobis for one session.

    Parameters
    ----------
    trim_buffers : bool
        If True (default), trim pre-start / post-reach buffers before
        computing crossnobis (used by superdiagonal, robustness).
        If False, keep buffers in (used by matrix heatmap).
        The cache key distinguishes the two variants.
    per_target : bool
        If True, compute crossnobis per target and return the
        target-averaged matrix as shape (1, T, T).
    per_target_stack : bool
        If True, compute crossnobis per target and return each
        target's fold-averaged matrix separately as shape
        (n_targets, T, T).  Implies per_target=True.

    Returns
    -------
    xnb_stack : ndarray, shape (n_folds, T, T) or (n_targets, T, T)
        Per-fold or per-target crossnobis matrices.
    dl : DataLoader
        The DataLoader used (needed by callers for target angles, etc.).
        Only created when computation is actually performed; ``None``
        when the result was loaded from cache.
    """
    if per_target_stack:
        per_target = True

    cache_path = _within_crossnobis_cache_path(
        session_data_name, unit_filter, window_config, trial_filter,
        per_target=per_target, suffix=cache_suffix)

    buf_tag = '' if trim_buffers else '_buffered'
    stack_tag = '_stack' if per_target_stack else ''
    fmt_key = f'crossnobis_matrices_{data_format}{buf_tag}{stack_tag}'

    # ── Try cache ────────────────────────────────────────────────────
    if not recompute and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        if fmt_key in cached:
            print(f'    Loaded cached {fmt_key}: {cache_path}')
            return cached[fmt_key], None

    # ── Compute ──────────────────────────────────────────────────────
    dl = utils_processing.DataLoader(
        data_dir, results_dir, session_data_name,
        unit_filter, None, window_config, trial_filter)
    dl.load_firing_rate_data()

    fr, *_ = dl.reformat_firing_rate_data(
        data_format,
        index_buffer=post_reach_idx,
        trial_length_filter_percentile=truncate_percentile)

    if trim_buffers:
        if data_format == 'truncate_end':
            fr = fr[:, :-post_reach_idx, :]
        elif data_format == 'truncate_front':
            fr = fr[:, pre_start_idx:, :]

    if per_target:
        target_angles = dl.get_target_angles()
        unique_targets = np.unique(target_angles)
        tgt_mats, tgt_min_T = [], np.inf
        for tgt in unique_targets:
            mask = target_angles == tgt
            fr_tgt = fr[mask]
            if fr_tgt.shape[0] < 3:
                continue
            _, mats_tgt = utils_processing.compute_crossnobis_matrix(
                fr_tgt, time_step=time_step)
            tgt_mats.append(np.mean(mats_tgt, axis=0))
            tgt_min_T = min(tgt_min_T, mats_tgt.shape[1])
        tgt_mats = [
            m_[:int(tgt_min_T), :int(tgt_min_T)] for m_ in tgt_mats]
        if per_target_stack:
            xnb_stack = np.stack(tgt_mats)  # (n_targets, T, T)
        else:
            xnb_stack = np.mean(np.stack(tgt_mats), axis=0)[np.newaxis]
    else:
        _, xnb_stack = utils_processing.compute_crossnobis_matrix(
            fr, time_step=time_step)

    # ── Save (merge into existing file) ──────────────────────────────
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
    else:
        cached = {}
    cached[fmt_key] = xnb_stack
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cached, f)
    print(f'    Saved {fmt_key}: {cache_path}')

    return xnb_stack, dl


def _compute_within_crossnobis_from_data(
        fr, time_step, per_target=False, target_angles=None):
    """Compute crossnobis from pre-loaded firing rate data (no caching).

    Used by robustness check for neuron-removed data.

    Returns
    -------
    xnb_stack : ndarray, shape (n_targets, T, T) if per_target else
                (n_folds, T, T)
    """
    if per_target:
        unique_targets = np.unique(target_angles)
        tgt_mats, tgt_min_T = [], np.inf
        for tgt in unique_targets:
            mask = target_angles == tgt
            fr_tgt = fr[mask]
            if fr_tgt.shape[0] < 3:
                continue
            _, mats_tgt = utils_processing.compute_crossnobis_matrix(
                fr_tgt, time_step=time_step)
            tgt_mats.append(np.mean(mats_tgt, axis=0))
            tgt_min_T = min(tgt_min_T, mats_tgt.shape[1])
        tgt_mats = [
            m_[:int(tgt_min_T), :int(tgt_min_T)] for m_ in tgt_mats]
        return np.stack(tgt_mats)  # (n_targets, T, T)
    else:
        _, xnb_stack = utils_processing.compute_crossnobis_matrix(
            fr, time_step=time_step)
        return xnb_stack


def _load_or_compute_cross_condition_crossnobis(
        session_data_name, unit_filter, window_config, trial_filters,
        data_format, time_step, truncate_percentile,
        pre_start_idx, post_reach_idx,
        per_target=False, recompute=False):
    """Load cached or compute cross-condition crossnobis for one session.

    Returns
    -------
    cross_avg : ndarray, shape (T1, T2)
        Averaged cross-condition crossnobis matrix.
    """
    cache_path = _cross_condition_crossnobis_cache_path(
        session_data_name, unit_filter, window_config, trial_filters,
        per_target=per_target)
    fmt_key = f'crossnobis_cross_{data_format}'

    # ── Try cache ────────────────────────────────────────────────────
    if not recompute and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        if fmt_key in cached:
            print(f'    Loaded cached cross-condition {data_format}: '
                  f'{cache_path}')
            return cached[fmt_key]

    # ── Compute ──────────────────────────────────────────────────────
    fr_data = {}
    target_idx_data = {}
    dl_data = {}
    for trial_filter in trial_filters:
        dl = utils_processing.DataLoader(
            data_dir, results_dir, session_data_name,
            unit_filter, None, window_config, trial_filter)
        dl.load_firing_rate_data()
        fr, *_ = dl.reformat_firing_rate_data(
            data_format,
            index_buffer=post_reach_idx,
            trial_length_filter_percentile=truncate_percentile)
        fr_data[trial_filter] = fr
        target_idx_data[trial_filter] = dl.get_target_indices()
        dl_data[trial_filter] = dl

    fr_1 = fr_data[trial_filters[0]]
    fr_2 = fr_data[trial_filters[1]]
    tgt_1 = target_idx_data[trial_filters[0]]
    tgt_2 = target_idx_data[trial_filters[1]]

    if per_target:
        unique_targets = np.intersect1d(
            np.unique(tgt_1), np.unique(tgt_2))
        # Fall back to angle-based matching when target indices don't overlap
        # (e.g. RadialGrid near vs far trials have non-overlapping indices)
        if len(unique_targets) == 0:
            tgt_1 = dl_data[trial_filters[0]].get_target_angles()
            tgt_2 = dl_data[trial_filters[1]].get_target_angles()
            unique_targets = np.intersect1d(
                np.unique(tgt_1), np.unique(tgt_2))
        session_cross = []
        for tgt in unique_targets:
            mask_1 = tgt_1 == tgt
            mask_2 = tgt_2 == tgt
            if mask_1.sum() < 2 or mask_2.sum() < 2:
                continue
            cross, _, _, _ = \
                utils_processing.compute_cross_condition_crossnobis_matrix(
                    fr_1[mask_1], fr_2[mask_2])
            session_cross.append(cross)
        if session_cross:
            cross_avg = np.mean(session_cross, axis=0)
        else:
            cross_avg = None
    else:
        cross_avg, _, _, _ = \
            utils_processing.compute_cross_condition_crossnobis_matrix(
                fr_1, fr_2)

    # ── Save ─────────────────────────────────────────────────────────
    if cross_avg is not None:
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
        else:
            cached = {}
        cached[fmt_key] = cross_avg
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(cached, f)
        print(f'    Saved cross-condition {data_format}: {cache_path}')

    return cross_avg


def _resample_to_common_time(traces_times, traces_values, time_step=0.01,
                             truncate_percentile=90):
    """Resample variable-length traces onto a common regular time grid.

    Parameters
    ----------
    traces_times  : list of 1-D arrays – time stamps per trial
    traces_values : list of 1-D arrays – values per trial (same lengths)
    time_step     : float – spacing of the output grid
    truncate_percentile : int or None
        If not None, remove the shortest (100-p)% of trials by duration,
        then truncate all remaining trials to the shortest surviving
        trial's duration.  Default 90 matches codebase convention.

    Returns
    -------
    common_times : 1-D array
    resampled    : 2-D array (n_kept_trials, n_common_times)
    """
    from scipy.interpolate import interp1d

    durations = np.array([t[-1] for t in traces_times])

    if truncate_percentile is not None:
        cutoff = np.percentile(durations, 100 - truncate_percentile)
        keep = durations >= cutoff
        traces_times  = [t for t, k in zip(traces_times, keep) if k]
        traces_values = [v for v, k in zip(traces_values, keep) if k]
        durations = durations[keep]

    t_max = durations.min()
    common_times = np.arange(0, t_max + time_step, time_step)
    resampled = np.full((len(traces_times), len(common_times)), np.nan)
    for i, (t, v) in enumerate(zip(traces_times, traces_values)):
        f = interp1d(t, v, kind='linear', bounds_error=False, fill_value=np.nan)
        resampled[i] = f(common_times)
    return common_times, resampled


def plot_behavioral_distances_to_target(
    session_data_names,
    unit_filter,
    window_config,
    data_format,
    trial_filters,
    aggregate='trials'):
    """
    Parameters
    ----------
    aggregate : str
        'trials'   – one thin line per trial (original behaviour)
        'mean_sem' – mean +/- SEM ribbon per condition
        'both'     – individual trials (faint) with mean +/- SEM overlay
    """

    task_name = session_data_names[0].split('_')[-1]

    fig, ax = plt.subplots(1, 1, figsize=(62.5*mm, 50*mm))

    for i_tf, trial_filter in enumerate(trial_filters):

        all_times  = []
        all_values = []

        for session_data_name in session_data_names:

            data_loader = utils_processing.DataLoader(
                data_dir, results_dir, session_data_name,
                unit_filter, None, window_config, trial_filter)
            data_loader.load_trial_ids_from_csv()
            data_loader.load_cursor_data()
            data_loader.remove_target_overlap(
                target_radius=session_target_radii[session_data_name])

            cursor_states, cursor_times = \
                data_loader.extract_cursor_states_and_times_without_alignment()
            target_positions = data_loader.get_target_positions()

            for i_trial in range(len(cursor_states)):
                cursor_positions = cursor_states[i_trial][:, 0:2]
                cursor_times_ = cursor_times[i_trial]
                target_position = target_positions[i_trial]
                distances = np.linalg.norm(
                    cursor_positions - target_position, axis=1)

                all_times.append(cursor_times_)
                all_values.append(distances)

        color = color_palettes[task_name][trial_filter][1]

        if aggregate == 'trials':
            for t, v in zip(all_times, all_values):
                ax.plot(t, v, color=color,
                        alpha=alpha_line_thin, lw=size_line_thin)
        elif aggregate in ('mean_sem', 'both'):
            if aggregate == 'both':
                for t, v in zip(all_times, all_values):
                    ax.plot(t, v, color=color,
                            alpha=0.10, lw=0.3)               # ← individual trial style
            common_t, resampled = _resample_to_common_time(
                all_times, all_values)
            mean = np.nanmean(resampled, axis=0)
            sem = np.nanstd(resampled, axis=0) / \
                np.sqrt(np.sum(~np.isnan(resampled), axis=0))
            ax.plot(common_t, mean, color=color,
                    alpha=alpha_line_thick, lw=size_line_thick,
                    solid_capstyle='round')
            if aggregate == 'mean_sem':
                ax.fill_between(common_t, mean - sem, mean + sem,
                                color=color, alpha=0.25)

    ax.set_xlabel('Time (s)', fontsize=7)
    ax.set_ylabel('Distance to target', fontsize=7)
    ax.set_xlim(0, 4.2)
    ax.set_ylim(0, 0.62)
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if len(session_data_names) > 3:
        session_data_names_str = str(len(session_data_names)) + '_sessions'
    else:
        session_data_names_str = str(session_data_names)

    res_name = '_'.join(map(str, [x for x in [
        'behavior_distances_to_target',
        session_data_names_str,
        unit_filter,
        data_format,
        trial_filters,
        aggregate] if x is not None]))

    save_path = os.path.join(vis_dir, res_name + '.pdf')
    plt.savefig(save_path, format="pdf", dpi=600,
                transparent=True, bbox_inches=None)
    plt.close()


def plot_behavioral_cursor_velocity(
    session_data_names,
    unit_filter,
    window_config,
    data_format,
    trial_filters,
    aggregate='trials'):
    """Plot cursor speed (magnitude of velocity) over time.

    Parameters
    ----------
    aggregate : str
        'trials'   – one thin line per trial
        'mean_sem' – mean +/- SEM ribbon per condition
        'both'     – individual trials (faint) with mean +/- SEM overlay
    """

    task_name = session_data_names[0].split('_')[-1]

    fig, ax = plt.subplots(1, 1, figsize=(62.5*mm, 50*mm))

    for i_tf, trial_filter in enumerate(trial_filters):

        all_times  = []
        all_values = []

        for session_data_name in session_data_names:

            data_loader = utils_processing.DataLoader(
                data_dir, results_dir, session_data_name,
                unit_filter, None, window_config, trial_filter)
            data_loader.load_trial_ids_from_csv()
            data_loader.load_cursor_data()
            data_loader.remove_target_overlap(
                target_radius=session_target_radii[session_data_name])

            cursor_states, cursor_times = \
                data_loader.extract_cursor_states_and_times_without_alignment()

            for i_trial in range(len(cursor_states)):
                cursor_velocities = cursor_states[i_trial][:, 2:4]
                cursor_times_ = cursor_times[i_trial]
                cursor_speed = np.linalg.norm(cursor_velocities, axis=1)

                all_times.append(cursor_times_)
                all_values.append(cursor_speed)

        color = color_palettes[task_name][trial_filter][1]

        if aggregate == 'trials':
            for t, v in zip(all_times, all_values):
                ax.plot(t, v, color=color,
                        alpha=alpha_line_thin, lw=size_line_thin)
        elif aggregate in ('mean_sem', 'both'):
            if aggregate == 'both':
                for t, v in zip(all_times, all_values):
                    ax.plot(t, v, color=color,
                            alpha=0.06, lw=0.3)               # ← individual trial style
            common_t, resampled = _resample_to_common_time(
                all_times, all_values)
            mean = np.nanmean(resampled, axis=0)
            sem = np.nanstd(resampled, axis=0) / \
                np.sqrt(np.sum(~np.isnan(resampled), axis=0))
            ax.plot(common_t, mean, color=color,
                    alpha=alpha_line_thick, lw=size_line_thick,
                    solid_capstyle='round')
            if aggregate == 'mean_sem':
                ax.fill_between(common_t, mean - sem, mean + sem,
                                color=color, alpha=0.25)

    ax.set_xlabel('Time (s)', fontsize=7)
    ax.set_ylabel('Cursor speed', fontsize=7)
    ax.set_xlim(0, 4.2)
    ax.set_ylim(0, 2.65)
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if len(session_data_names) > 3:
        session_data_names_str = str(len(session_data_names)) + '_sessions'
    else:
        session_data_names_str = str(session_data_names)

    res_name = '_'.join(map(str, [x for x in [
        'behavior_cursor_velocity',
        session_data_names_str,
        unit_filter,
        data_format,
        trial_filters,
        aggregate] if x is not None]))

    save_path = os.path.join(vis_dir, res_name + '.pdf')
    plt.savefig(save_path, format="pdf", dpi=600,
                transparent=True, bbox_inches=None)
    plt.close()


def plot_behavioral_target_acquisition_times(
    session_data_names,
    unit_filter,
    window_config,
    data_format,
    trial_filters,
    max_cols=6,
    show_points=False):

    task_name = session_data_names[0].split('_')[-1]
    n_sessions = len(session_data_names)

    target_acquisition_times_df_list = []

    for i_tf, trial_filter in enumerate(trial_filters):

        for i_session, session_data_name in enumerate(session_data_names):

            ## Load data
            data_loader = utils_processing.DataLoader(
                data_dir,
                results_dir,
                session_data_name,
                unit_filter,
                None,
                window_config,
                trial_filter)

            data_loader.load_firing_rate_data()
            data_loader.load_cursor_data()
            data_loader.remove_target_overlap(target_radius=session_target_radii[session_data_name])

            _, cursor_times = data_loader.extract_cursor_states_and_times_without_alignment()
            n_trials = len(cursor_times)

            target_acquisition_times = [cursor_times[i_trial][-1] for i_trial in range(n_trials)]

            target_acquisition_times_trials_df = pd.DataFrame({
                'session': ['Session ' + str(i_session + 1)] * n_trials,
                'trial_filter': [trial_filter] * n_trials,
                'target_acquisition_time': target_acquisition_times})

            target_acquisition_times_df_list.append(target_acquisition_times_trials_df)

    target_acquisition_times_df = pd.concat(target_acquisition_times_df_list, ignore_index=True)

    ## Compute statistics
    # Group data by trial filter
    if task_name == 'RadialGrid':
        fast_trials = target_acquisition_times_df[target_acquisition_times_df['trial_filter'] == 'near']['target_acquisition_time'].values
        slow_trials = target_acquisition_times_df[target_acquisition_times_df['trial_filter'] == 'far']['target_acquisition_time'].values
    else:
        fast_trials = target_acquisition_times_df[target_acquisition_times_df['trial_filter'] == 'fast']['target_acquisition_time'].values
        slow_trials = target_acquisition_times_df[target_acquisition_times_df['trial_filter'] == 'slow']['target_acquisition_time'].values

    # Compute means and standard deviations
    mean_fast = np.mean(fast_trials)
    mean_slow = np.mean(slow_trials)
    std_fast = np.std(fast_trials, ddof=1)
    std_slow = np.std(slow_trials, ddof=1)

    print('Fast trials: ')
    print('\tmean: ', mean_fast)
    print('\tstd: ', std_fast)
    print('Slow trials: ')
    print('\tmean: ', mean_slow)
    print('\tstd: ', std_slow)

    # Sample sizes
    n_fast = len(fast_trials)
    n_slow = len(slow_trials)

    # Compute mean difference
    mean_diff = mean_slow - mean_fast

    # Pooled standard error
    se_diff = np.sqrt((std_fast**2 / n_fast) + (std_slow**2 / n_slow))

    # Degrees of freedom (Welch-Satterthwaite equation)
    df = ((std_fast**2 / n_fast + std_slow**2 / n_slow)**2) / \
         (((std_fast**2 / n_fast)**2 / (n_fast - 1)) + ((std_slow**2 / n_slow)**2 / (n_slow - 1)))

    # Compute t critical value for 95% CI
    alpha = 0.05
    t_crit = t.ppf(1 - alpha / 2, df=df)

    # Margin of error
    margin_of_error = t_crit * se_diff

    # Confidence interval
    ci_lower = mean_diff - margin_of_error
    ci_upper = mean_diff + margin_of_error

    # Mann-Whitney U test
    stat_u, p_u = mannwhitneyu(fast_trials, slow_trials, alternative='two-sided')

    # Cohen's d (independent samples, pooled SD)
    pooled_sd = np.sqrt(((n_fast - 1) * std_fast**2 + (n_slow - 1) * std_slow**2)
                        / (n_fast + n_slow - 2))
    cohens_d = (mean_fast - mean_slow) / pooled_sd

    # Output the results
    print(f"Mean difference (fast - slow): {mean_fast - mean_slow:.3f}")
    print(f"Mean difference (slow - fast): {mean_diff:.3f}")
    print(f"95% CI: ({ci_lower:.3f}, {ci_upper:.3f})")
    print(f"Mann-Whitney U: {stat_u:.0f}   P-value: {p_u:.3g}")
    print(f"Cohen's d (fast - slow): {cohens_d:.3f}")

    # Use task-specific colors from vis_config (index [1] = mid tone)
    palette = {tf: color_palettes[task_name][tf][1] for tf in trial_filters}

    # Determine grid layout
    n_cols = min(n_sessions, max_cols)
    n_rows = int(np.ceil(n_sessions / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(20 * mm * n_cols, 55 * mm * n_rows),
                             sharey=True, squeeze=False)

    for i_session in range(n_sessions):
        row = i_session // n_cols
        col = i_session % n_cols
        ax = axes[row, col]
        session_label = f'Session {i_session + 1}'
        session_df = target_acquisition_times_df[
            target_acquisition_times_df['session'] == session_label]

        sns.violinplot(
            ax=ax, data=session_df,
            x='session', y='target_acquisition_time',
            hue='trial_filter', split=True,
            cut=0, inner=None,
            palette=palette,
            linewidth=0.25, saturation=1, alpha=0.8)

        # Remove edge colors from violin bodies
        for art in ax.findobj(mcoll.PolyCollection):
            art.set_edgecolor('none')

        # Overlay individual data points
        if show_points:
            for i_tf, tf in enumerate(trial_filters):
                tf_data = session_df[session_df['trial_filter'] == tf]['target_acquisition_time'].values
                # Jitter x: left half (i_tf==0) -> negative offset, right half -> positive
                x_center = 0  # single x position per session
                jitter = np.random.default_rng(seed=i_session * 10 + i_tf).uniform(
                    0.02, 0.15, size=len(tf_data))
                if i_tf == 0:
                    x_pos = x_center - jitter
                else:
                    x_pos = x_center + jitter
                ax.scatter(x_pos, tf_data, s=0.5, color=palette[tf],
                           alpha=0.5, edgecolors='none', rasterized=False)

        # Overlay median + IQR as point + error bar for each condition
        for i_tf, tf in enumerate(trial_filters):
            tf_data = session_df[session_df['trial_filter'] == tf]['target_acquisition_time'].values
            if len(tf_data) == 0:
                continue
            median = np.median(tf_data)
            q25 = np.percentile(tf_data, 25)
            q75 = np.percentile(tf_data, 75)
            x_center = 0
            x_offset = -0.05 if i_tf == 0 else 0.05
            ax.vlines(x_center + x_offset, q25, q75, colors='black',
                      linewidth=2, capstyle='round', zorder=5)
            ax.plot(x_center + x_offset, median, 'o',
                    markerfacecolor='white', markeredgecolor='black',
                    markeredgewidth=0, markersize=1.5, zorder=6)

        ax.set_ylim(0, 4)
        ax.set_xlabel(None)
        ax.set_xticks([0])
        ax.set_xticklabels([session_label], rotation=30, ha='right', fontsize=7)
        sns.despine(ax=ax, top=True, right=True,
                    left=(col > 0), bottom=False)
        if col == 0:
            ax.set_ylabel('Target acquisition time (s)', fontsize=7)
        else:
            ax.set_ylabel(None)
            ax.tick_params(axis='y', left=False, labelleft=False)

        ax.tick_params(axis='both', labelsize=5)
        if ax.get_legend() is not None:
            ax.legend_.remove()

    # Hide unused axes
    for i_ax in range(n_sessions, n_rows * n_cols):
        row = i_ax // n_cols
        col = i_ax % n_cols
        axes[row, col].set_visible(False)

    fig.tight_layout()
    # plt.subplots_adjust(wspace=0.05, bottom=0.2)

    ## Write image
    if len(session_data_names) > 3:
        session_data_names_str = str(len(session_data_names)) + '_sessions'
    else:
        session_data_names_str = str(session_data_names)

    res_name = '_'.join(map(str, [x for x in [
        'behavior_target_acquisition_times',
        session_data_names_str,
        unit_filter,
        data_format,
        trial_filters] if x is not None]))

    save_path = os.path.join(vis_dir, res_name + '.pdf')

    plt.savefig(save_path, format="pdf", dpi=600, transparent=True, bbox_inches=None)
    plt.close()


def plot_behavioral_target_acquisition_times_pooled(
    session_data_names,
    unit_filter,
    window_config,
    data_format,
    trial_filters):

    task_name = session_data_names[0].split('_')[-1]

    target_acquisition_times_df_list = []

    for i_tf, trial_filter in enumerate(trial_filters):
        for i_session, session_data_name in enumerate(session_data_names):
            data_loader = utils_processing.DataLoader(
                data_dir,
                results_dir,
                session_data_name,
                unit_filter,
                None,
                window_config,
                trial_filter)

            data_loader.load_firing_rate_data()
            data_loader.load_cursor_data()
            data_loader.remove_target_overlap(target_radius=session_target_radii[session_data_name])

            _, cursor_times = data_loader.extract_cursor_states_and_times_without_alignment()
            n_trials = len(cursor_times)
            target_acquisition_times = [cursor_times[i_trial][-1] for i_trial in range(n_trials)]

            target_acquisition_times_df_list.append(pd.DataFrame({
                'session': ['Session ' + str(i_session + 1)] * n_trials,
                'trial_filter': [trial_filter] * n_trials,
                'target_acquisition_time': target_acquisition_times}))

    target_acquisition_times_df = pd.concat(target_acquisition_times_df_list, ignore_index=True)

    ## Compute and print statistics
    if task_name == 'RadialGrid':
        fast_trials = target_acquisition_times_df[target_acquisition_times_df['trial_filter'] == 'near']['target_acquisition_time'].values
        slow_trials = target_acquisition_times_df[target_acquisition_times_df['trial_filter'] == 'far']['target_acquisition_time'].values
    else:
        fast_trials = target_acquisition_times_df[target_acquisition_times_df['trial_filter'] == 'fast']['target_acquisition_time'].values
        slow_trials = target_acquisition_times_df[target_acquisition_times_df['trial_filter'] == 'slow']['target_acquisition_time'].values

    mean_fast, mean_slow = np.mean(fast_trials), np.mean(slow_trials)
    std_fast, std_slow = np.std(fast_trials, ddof=1), np.std(slow_trials, ddof=1)
    n_fast, n_slow = len(fast_trials), len(slow_trials)

    print('Fast trials: ')
    print('\tmean: ', mean_fast)
    print('\tstd: ', std_fast)
    print('Slow trials: ')
    print('\tmean: ', mean_slow)
    print('\tstd: ', std_slow)

    mean_diff = mean_fast - mean_slow
    se_diff = np.sqrt((std_fast**2 / n_fast) + (std_slow**2 / n_slow))
    df = ((std_fast**2 / n_fast + std_slow**2 / n_slow)**2) / \
         (((std_fast**2 / n_fast)**2 / (n_fast - 1)) + ((std_slow**2 / n_slow)**2 / (n_slow - 1)))
    t_crit = t.ppf(1 - 0.05 / 2, df=df)
    margin_of_error = t_crit * se_diff
    ci_lower = mean_diff - margin_of_error
    ci_upper = mean_diff + margin_of_error

    # Mann-Whitney U test
    stat_u, p_u = mannwhitneyu(fast_trials, slow_trials, alternative='two-sided')

    # Cohen's d (independent samples, pooled SD)
    pooled_sd = np.sqrt(((n_fast - 1) * std_fast**2 + (n_slow - 1) * std_slow**2)
                        / (n_fast + n_slow - 2))
    cohens_d = mean_diff / pooled_sd

    print(f"Mean difference (fast - slow): {mean_diff:.3f}")
    print(f"95% CI: ({ci_lower:.3f}, {ci_upper:.3f})")
    print(f"Mann-Whitney U: {stat_u:.0f}   P-value: {p_u:.3g}")
    print(f"Cohen's d (fast - slow): {cohens_d:.3f}")

    # Colors
    palette = {tf: color_palettes[task_name][tf][1] for tf in trial_filters}

    # Layout: single axis, 4 horizontal bands
    # Condition 1 (trial_filters[0]): dots on left, half-violin center-left
    # Condition 2 (trial_filters[1]): half-violin center-right, dots on right
    fig, ax = plt.subplots(1, 1, figsize=(40 * mm, 50 * mm))

    tf0, tf1 = trial_filters[0], trial_filters[1]
    data0 = target_acquisition_times_df[target_acquisition_times_df['trial_filter'] == tf0]['target_acquisition_time'].values
    data1 = target_acquisition_times_df[target_acquisition_times_df['trial_filter'] == tf1]['target_acquisition_time'].values

    # --- Half violins via manual KDE ---
    from scipy.stats import gaussian_kde
    y_grid = np.linspace(0, 4, 300)

    # tf0 (fast/near) on left (-x), tf1 (slow/far) on right (+x)
    for data, tf, side in [(data0, tf0, 'left'), (data1, tf1, 'right')]:
        kde = gaussian_kde(data, bw_method=0.2)
        density = kde(y_grid)
        density = density / density.max() * 0.3

        if side == 'left':
            # Half-violin opens toward center (positive x direction)
            ax.fill_betweenx(y_grid, -density, 0, color=palette[tf],
                             alpha=0.8, edgecolor='none')
        else:
            # Half-violin opens toward center (negative x direction)
            ax.fill_betweenx(y_grid, 0, density, color=palette[tf],
                             alpha=0.8, edgecolor='none')

    # --- Individual dots ---
    dot_spread = 0.15
    rng = np.random.default_rng(seed=42)

    # Condition 1 dots: to the left of its half-violin
    jitter0 = rng.uniform(-0.32 - dot_spread, -0.32, size=len(data0))
    ax.scatter(jitter0, data0, s=1, color=palette[tf0],
               alpha=0.5, edgecolors='none', rasterized=False)

    # Condition 2 dots: to the right of its half-violin
    jitter1 = rng.uniform(0.32, 0.32 + dot_spread, size=len(data1))
    ax.scatter(jitter1, data1, s=1, color=palette[tf1],
               alpha=0.5, edgecolors='none', rasterized=False)

    # --- Median + IQR ---
    for data, tf, x_offset in [(data0, tf0, -0.05), (data1, tf1, 0.05)]:
        median = np.median(data)
        q25 = np.percentile(data, 25)
        q75 = np.percentile(data, 75)
        ax.vlines(x_offset, q25, q75, colors='black',
                  linewidth=1.5, capstyle='round', zorder=5)
        ax.plot(x_offset, median, 'o',
                markerfacecolor='white', markeredgecolor='black',
                markeredgewidth=0.5, markersize=2.5, zorder=6)

    ax.set_ylim(0, 4)
    ax.set_xlim(-0.55, 0.55)

    # X-axis labels under each violin center
    if task_name == 'RadialGrid':
        label0, label1 = 'near', 'far'
    else:
        label0, label1 = 'ballistic', 'sustained'
    ax.set_xticks([-0.25, 0.25])
    ax.set_xticklabels([label0, label1], fontsize=5)

    ax.set_ylabel('Target acquisition time (s)', fontsize=7)
    ax.tick_params(axis='y', labelsize=5)
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()

    ## Write image
    if len(session_data_names) > 3:
        session_data_names_str = str(len(session_data_names)) + '_sessions'
    else:
        session_data_names_str = str(session_data_names)

    res_name = '_'.join(map(str, [x for x in [
        'behavior_target_acquisition_times_pooled',
        session_data_names_str,
        unit_filter,
        data_format,
        trial_filters] if x is not None]))

    save_path = os.path.join(vis_dir, res_name + '.pdf')
    plt.savefig(save_path, format="pdf", dpi=600, transparent=True, bbox_inches=None)
    plt.close()


def plot_cursor_trajectories(
    session_data_name,
    unit_filter,
    input_unit_filter,
    window_config,
    data_format,
    trial_filter,
    train_test_option,
    random_state,
    n_continuous_states,
    n_discrete_states,
    n_iters,
    model_type,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    show_target_positions=True,
    discrete_state_overlay=False):

    task = session_data_name.split('_')[-1]

    data_loader = utils_processing.DataLoader(
        data_dir,
        results_dir,
        session_data_name,
        unit_filter,
        input_unit_filter,
        window_config,
        trial_filter)

    data_loader.load_firing_rate_data()
    data_loader.load_cursor_data()
    data_loader.remove_target_overlap(target_radius=session_target_radii[session_data_name])

    cursor_states_all, cursor_times = data_loader.extract_cursor_states_and_times_without_alignment()

    _, _, _, _, _, trial_keep_mask = data_loader.reformat_firing_rate_data(
        data_format, 
        trial_length_filter_percentile=90)

    cursor_states_all = [cursor_states for cursor_states, keep_mask in zip(cursor_states_all, trial_keep_mask) if keep_mask]
    n_trials = len(cursor_states_all)

    target_ids = data_loader.get_target_ids()
    # target_positions = data_loader.get_target_positions()
    
    ## Plot cursor trajectories
    fig, axs = plt.subplots(1, 1, figsize=(45*mm, 45*mm))
    cmap = color_maps[task][trial_filter]

    if discrete_state_overlay:

        print('Continuous States: ' + str(n_continuous_states) + ', Discrete States: ' + str(n_discrete_states))

        ## Load SLDS results
        model_results_dir = data_loader.get_model_result_dir(
            # time_offset=time_offset,
            data_format=data_format,
            train_test=train_test_option,
            model_type=model_type,
            dynamics_class=dynamics_class,
            emission_class=emission_class,
            init_type=init_type,
            subspace_type=subspace_type,
            alpha=alpha,
            check_existence=True)

        ## Read SLDS processed data
        if model_type in ['LDS']:

            ## Omit discrete states for LDS
            model_save_name = '_'.join(map(str, [x for x in [
                'r' + str(random_state),
                's' + str(n_continuous_states),
                'i' + str(n_iters)]]))
        else:
            model_save_name = '_'.join(map(str, [x for x in [
                'r' + str(random_state),
                's' + str(n_continuous_states),
                'd' + str(n_discrete_states),
                'i' + str(n_iters)]]))

        res_save_path = os.path.join(model_results_dir, model_save_name) + '.pkl'

        with open(res_save_path, 'rb') as f:
            res_SLDS = pickle.load(f)

        discrete_states = res_SLDS['test_discrete_states']

        ## Reorder discrete states
        discrete_states, _, _ = utils_vis.reorder_discrete_states(discrete_states, n_discrete_states)

        for i_trial in range(n_trials):
            utils_vis.add_color_graded_trajectory(
                axs, 
                cursor_states_all[i_trial][:, 0:2], 
                cmap,
                linewidth=size_line_thin,
                alpha=alpha_line_thick,
                t=discrete_states[i_trial], 
                t_max=1)
    
    else:
        for i_trial in range(n_trials):
            target_id = target_ids[i_trial] - 1
            axs.plot(
                cursor_states_all[i_trial][:, 0], 
                cursor_states_all[i_trial][:, 1], 
                color=target_color_palette_8[target_id - 1], 
                alpha=alpha_line, 
                lw=size_line_thin)
            
            ## Plotting the cursor at each time point
            # axs.scatter(
            #     cursor_states_all[i_trial][:, 0],
            #     cursor_states_all[i_trial][:, 1],
            #     color=target_color_palette_8[target_id - 1],
            #     alpha=alpha_line,
            #     marker='o',
            #     s=5)
    
    if show_target_positions:
        for i_target in range(8):
            axs.add_patch(Circle(
                (np.cos(np.pi / 4 * i_target) * 0.4, np.sin(np.pi / 4 * i_target) * 0.4),
                # (target_positions[i_target, 0], target_positions[i_target, 1]),
                radius=0.1,
                fill=False,
                linestyle='--',
                edgecolor='black',
                linewidth=size_line_thin,
                alpha=alpha_line_thin))
        
        axs.add_patch(Circle(
            (0, 0),
            radius=0.1,
            fill=False,
            linestyle='--',
            edgecolor='black',
            linewidth=size_line_thin,
            alpha=alpha_line_thin))
        
    # custom_lines = [
    #     Line2D([0], [0], color=color_palette(0.0), lw=size_line, alpha=alpha_line, linestyle='-'),
    #     Line2D([0], [0], color=color_palette(1.0), lw=size_line, alpha=alpha_line, linestyle='-')]

    # legend_labels = [
    #     'Transient Phase',
    #     'Steady Phase']
    
    # ## Create legend with custom elements
    # plt.legend(
    #     custom_lines, 
    #     legend_labels, 
    #     fancybox=False, 
    #     shadow=False,
    #     fontsize=5)
        
            
    x_min = np.min([np.min(cursor_states_all[i][:, 0]) for i in range(n_trials)])
    x_max = np.max([np.max(cursor_states_all[i][:, 0]) for i in range(n_trials)])
    y_min = np.min([np.min(cursor_states_all[i][:, 1]) for i in range(n_trials)])
    y_max = np.max([np.max(cursor_states_all[i][:, 1]) for i in range(n_trials)])
    x_range = x_max - x_min
    y_range = y_max - y_min


    ## Set titles
    # axs[i_discrete_states, i_continuous_states].set_title('# of Continuous States: ' + str(n_continuous_states) + ', # of Discrete States: ' + str(n_discrete_states))

    ## Set axis labels
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_xticklabels([])
    axs.set_yticklabels([])
    # axs.set_xlim(x_min - x_range * 0.2, x_max + x_range * 0.2)
    # axs.set_ylim(y_min - y_range * 0.2, y_max + y_range * 0.2)
    axs.set_xlim(-0.55, 0.55)
    axs.set_ylim(-0.55, 0.55)

    # Remove the top and right spines
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['left'].set_visible(False)
    axs.spines['bottom'].set_visible(False)

    ## Set big title
    # fig.suptitle(session_data_name)

    plt.tight_layout()

    ## Write image
    session_vis_dir = os.path.join(vis_dir, session_data_name)

    if not os.path.isdir(session_vis_dir):
        os.makedirs(session_vis_dir)

    if discrete_state_overlay:
        res_name = '_'.join(map(str, [x for x in [
            'cursor_trajectories',
            unit_filter,
            input_unit_filter,
            window_config,
            time_offset,
            data_format,
            trial_filter,
            train_test_option,
            random_state,
            n_continuous_states,
            n_discrete_states,
            n_iters,
            model_type,
            dynamics_class,
            emission_class,
            init_type,
            subspace_type,
            alpha] if x is not None]))
    else:
        res_name = '_'.join(map(str, [x for x in [
            'cursor_trajectories',
            unit_filter,
            input_unit_filter,
            window_config,
            time_offset,
            data_format,
            trial_filter] if x is not None]))
    
    if show_target_positions:
        res_name += '_tp'

    save_path = os.path.join(session_vis_dir, res_name + '.pdf')

    plt.savefig(save_path, dpi=600, transparent=True, bbox_inches=None)
    plt.close()


def plot_cursor_trajectories_simple(
    session_data_name,
    unit_filter,
    data_format,
    show_average_trajectories=True,
    show_example_trajectories=False,
    show_target_positions=True):

    line_styles = ['--', '-']

    ## Plot cursor trajectories
    fig, axs = plt.subplots(1, 1, figsize=(50*mm, 50*mm))

    for i_tf, trial_filter in enumerate(trial_filters):

        data_loader = utils_processing.DataLoader(
            data_dir,
            results_dir,
            session_data_name,
            unit_filter,
            None,
            window_config,
            trial_filter)

        data_loader.load_firing_rate_data()
        data_loader.load_cursor_data()
        data_loader.remove_target_overlap(target_radius=session_target_radii[session_data_name])

        cursor_states_all, cursor_times = data_loader.extract_cursor_states_and_times_without_alignment()
        n_trials = len(cursor_times)

        target_ids = data_loader.get_target_ids()
        # target_positions = data_loader.get_target_positions()

        ## Plot trajectories averaged by target
        if show_average_trajectories:

            target_ids_unique = np.unique(target_ids)

            ## Resample continuous states to have the same length
            if data_format is None:
                cursor_states_all, _, _ = utils_processing.resample_emissions(cursor_states_all)

            cursor_states_all = np.array(cursor_states_all)

            for target_id in target_ids_unique:
                target_filter = (target_ids == target_id)
                cursor_states_avg = np.mean(cursor_states_all[target_filter, :, :], axis=0)                

                axs.plot(
                    cursor_states_avg[:, 0], 
                    cursor_states_avg[:, 1],
                    color=target_color_palette_8[target_id - 1],
                    lw=size_line,
                    alpha=alpha_line,
                    linestyle=line_styles[i_tf])


        elif show_example_trajectories:

            target_ids_unique = np.unique(target_ids)

            ## Resample continuous states to have the same length
            for target_id in target_ids_unique:
                target_filter = (target_ids == target_id)
                example_target_id = np.where(target_filter)[0][1]
                cursor_states_example = cursor_states_all[example_target_id][:, :]

                axs.plot(
                    cursor_states_example[:, 0], 
                    cursor_states_example[:, 1],
                    color=target_color_palette_8[target_id - 1],
                    lw=size_line,
                    alpha=alpha_line,
                    linestyle=line_styles[i_tf])


        else:
            for i_trial in range(n_trials):
                target_id = target_ids[i_trial] - 1
                axs.plot(
                    cursor_states_all[i_trial][:, 0], 
                    cursor_states_all[i_trial][:, 1], 
                    color=target_color_palette_8[target_id - 1], 
                    alpha=alpha_line, 
                    lw=size_line_thin)

                ## Plotting the cursor at each time point
                # axs.scatter(
                #     cursor_states_all[i_trial][:, 0],
                #     cursor_states_all[i_trial][:, 1],
                #     color=target_color_palette_8[target_id - 1],
                #     alpha=alpha_line,
                #     marker='o',
                #     s=5)

    
    if show_target_positions:
        for i_target in range(8):
            axs.add_patch(Circle(
                (np.cos(np.pi / 4 * i_target) * 0.4, np.sin(np.pi / 4 * i_target) * 0.4),
                # (target_positions[i_target, 0], target_positions[i_target, 1]),
                radius=0.1,
                fill=False,
                linestyle='--',
                edgecolor='black',
                linewidth=size_line_thin,
                alpha=alpha_line_thin))
        
        axs.add_patch(Circle(
            (0, 0),
            radius=0.1,
            fill=False,
            linestyle='--',
            edgecolor='black',
            linewidth=size_line_thin,
            alpha=alpha_line_thin))
        
    
    ## Set legends
    custom_lines = [
        Line2D([0], [0], color='black', lw=size_line, alpha=alpha_line, linestyle='-'),
        Line2D([0], [0], color='black', lw=size_line, alpha=alpha_line, linestyle='--')]

    legend_labels = [
        'Ballistic',
        'Sustained']
    
    ## Create legend with custom elements
    # plt.legend(
    #     custom_lines, 
    #     legend_labels, 
    #     fancybox=False, 
    #     shadow=False,
    #     fontsize=5)

    
    x_min = np.min([np.min(cursor_states_all[i][:, 0]) for i in range(n_trials)])
    x_max = np.max([np.max(cursor_states_all[i][:, 0]) for i in range(n_trials)])
    y_min = np.min([np.min(cursor_states_all[i][:, 1]) for i in range(n_trials)])
    y_max = np.max([np.max(cursor_states_all[i][:, 1]) for i in range(n_trials)])
    x_range = x_max - x_min
    y_range = y_max - y_min


    ## Set titles
    # axs[i_discrete_states, i_continuous_states].set_title('# of Continuous States: ' + str(n_continuous_states) + ', # of Discrete States: ' + str(n_discrete_states))

    ## Set axis labels
    # axs.set_xlabel('X Position', fontsize=label_fontsize)
    # axs.set_ylabel('Y Position', fontsize=label_fontsize)
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_xticklabels([])
    axs.set_yticklabels([])
    # axs.set_xlim(x_min - x_range * 0.2, x_max + x_range * 0.2)
    # axs.set_ylim(y_min - y_range * 0.2, y_max + y_range * 0.2)
    axs.set_xlim(-0.55, 0.55)
    axs.set_ylim(-0.55, 0.55)

    # Remove the top and right spines
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['left'].set_visible(False)
    axs.spines['bottom'].set_visible(False)

    ## Set big title
    # fig.suptitle(session_data_name)

    plt.tight_layout()

    ## Write image
    session_vis_dir = os.path.join(vis_dir, session_data_name)

    if not os.path.isdir(session_vis_dir):
        os.makedirs(session_vis_dir)

    res_name = '_'.join(map(str, [x for x in [
        'cursor_trajectories_simple',
        unit_filter,
        window_config,
        time_offset,
        data_format] if x is not None]))
    
    if show_average_trajectories:
        res_name += '_avg'
    elif show_example_trajectories:
        res_name += '_ex'
    if show_target_positions:
        res_name += '_tp'

    save_path = os.path.join(session_vis_dir, res_name + '.pdf')

    plt.savefig(save_path, format="pdf", dpi=600, transparent=True, bbox_inches=None)
    plt.close()


def plot_discrete_states_over_time_single_session(
    session_data_name,
    unit_filter,
    input_unit_filter,
    data_format,
    random_state,
    n_continuous_states,
    n_discrete_states,
    n_iters,
    model_type,
    dynamics_class,
    emission_class,
    init_type,
    alpha,
    t_start=0):

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
    
    fast_model_results_dir, slow_model_results_dir = data_loader.get_model_result_dirs(
        time_offset=time_offset,
        train_test='same_speed',
        # train_test='joint',
        data_format=data_format,
        model_type=model_type,
        dynamics_class=dynamics_class,
        emission_class=emission_class,
        init_type=init_type,
        alpha=alpha,
        check_existence=True)
    
    print(slow_model_results_dir)
    print(fast_model_results_dir)
    
    color_palette = trial_filter_colors_by_discrete_state

    # fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    fig, axs = plt.subplots(1, 2, figsize=(20, 4))
    
    ## Read SLDS processed data
    if model_type in ['LDS']:

        ## Omit discrete states for LDS
        model_save_name = '_'.join(map(str, [x for x in [
            'r' + str(random_state),
            's' + str(n_continuous_states),
            'i' + str(n_iters)]]))
    else:
        model_save_name = '_'.join(map(str, [x for x in [
            'r' + str(random_state),
            's' + str(n_continuous_states),
            'd' + str(n_discrete_states),
            'i' + str(n_iters)]]))

    slow_res_save_path = os.path.join(slow_model_results_dir, model_save_name) + '.pkl'
    fast_res_save_path = os.path.join(fast_model_results_dir, model_save_name) + '.pkl'

    try:
        with open(slow_res_save_path, 'rb') as f:
            res_SLDS_slow = pickle.load(f)

        with open(fast_res_save_path, 'rb') as f:
            res_SLDS_fast = pickle.load(f)
    except:
        print('Model results not found')
        print('Slow:', slow_res_save_path)
        print('Fast:', fast_res_save_path)
        return
    
    ## Organize data by trial filter
    data_by_trial_filter = {
        'fast': {
        # 'near': {
            'continuous_states': res_SLDS_fast['test_continuous_states'],
            'discrete_states'  : res_SLDS_fast['test_discrete_states'],
            'model'            : res_SLDS_fast['model']
        },
        'slow': {
        # 'far': {
            'continuous_states': res_SLDS_slow['test_continuous_states'],
            'discrete_states'  : res_SLDS_slow['test_discrete_states'],
            'model'            : res_SLDS_slow['model']
        }
    }

    max_trial_length_all = 0

    initial_durations_all = []

    ## Get the max trial length for both slow and fast trials
    for i_tf, trial_filter in enumerate(data_by_trial_filter.keys()):
        data = data_by_trial_filter[trial_filter]
        discrete_states = data['discrete_states']
        trial_lengths = [len(discrete_states_) for discrete_states_ in discrete_states]
        max_trial_length = np.max(trial_lengths)
        max_trial_length_all = max(max_trial_length_all, max_trial_length)

    for i_tf, trial_filter in enumerate(data_by_trial_filter.keys()):
        data = data_by_trial_filter[trial_filter]
        discrete_states = data['discrete_states']
        n_trials = len(discrete_states)
        trial_lengths = [len(discrete_states_) for discrete_states_ in discrete_states]
        min_trial_length = np.min(trial_lengths)

        ## Reorder discrete states
        discrete_states, _, _ = utils_vis.reorder_discrete_states(discrete_states, n_discrete_states)

        discrete_state_matrix = np.full((n_trials, max_trial_length_all), -1)

        for i_trial, trial_length in enumerate(trial_lengths):
            discrete_state_matrix[i_trial, :trial_length] = discrete_states[i_trial]

        initial_durations = utils_vis.find_initial_zero_durations(discrete_state_matrix)
        initial_durations_all.append(initial_durations)

        ## Define x axis
        # x = np.arange(min_trial_length) * config.time_step
        x = np.arange(max_trial_length_all) * config.time_step + t_start
        # x = np.arange(min_trial_length)

        bar_width = config.time_step * 0.8

        ## Set base for stacked bar plot
        base = np.zeros(max_trial_length_all)

        for i_discrete_state in range(n_discrete_states):
            discrete_state_count = np.sum(discrete_state_matrix == i_discrete_state, axis=0)
            axs[i_tf].bar(
                x, 
                discrete_state_count, 
                bottom=base, 
                color=color_palette[trial_filter][i_discrete_state],
                width=bar_width,
                align='edge')

            base += discrete_state_count
        
        ## Set axis limits
        axs[i_tf].set_ylim(0, n_trials)
        axs[i_tf].set_xlim(0 + t_start, max_trial_length_all * config.time_step + t_start)
        xticks = [t_start] + np.arange(0.5, max_trial_length_all * config.time_step, 0.5).tolist()
        axs[i_tf].set_xticks(xticks)
        # axs[i_tf].set_xtickslabels(xticks)

        ## Set axis labels
        axs[i_tf].set_xlabel('Time (s)', fontsize=label_fontsize)
        axs[i_tf].set_ylabel(trial_filter_name_conversion[trial_filter] + ' Trial Count', fontsize=label_fontsize)


    ## Statistical test
    stat, p_value = mannwhitneyu(initial_durations_all[0], initial_durations_all[1], alternative='two-sided')
    print('Fast trial mean:', np.mean(initial_durations_all[0]), 'Slow trial mean:', np.mean(initial_durations_all[1]))
    print("Mann-Whitney U Statistic:", stat)
    print("P-Value:", p_value)

    ## Set big title
    # fig.suptitle(session_data_name)

    plt.tight_layout()

    ## Write image
    session_vis_dir = os.path.join(vis_dir, session_data_name)

    if not os.path.isdir(session_vis_dir):
        os.makedirs(session_vis_dir)

    res_name = '_'.join(map(str, [x for x in [
        'discrete_states_over_time',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        random_state,
        n_continuous_states,
        n_discrete_states,
        n_iters,
        model_type,
        dynamics_class,
        emission_class,
        init_type,
        alpha] if x is not None]))

    save_path = os.path.join(session_vis_dir, res_name + '.pdf')

    plt.savefig(save_path, dpi=600, transparent=True)
    plt.close()


def plot_discrete_states_over_time(
    session_data_names,
    unit_filter,
    input_unit_filter,
    window_config,
    time_step,
    data_format,
    trial_filters,
    train_test_option,
    random_state,
    n_continuous_states,
    n_discrete_states,
    n_iters,
    model_type,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    visual_delay_time=0):

    task_name = session_data_names[0].split('_')[-1]
    color_palette = discrete_state_colors[task_name]

    # ── 1.  Make the two rows share their x-axis  ────────────────────────────────
    fig, axs = plt.subplots(
        2, 1,
        figsize=(90 * mm, 45 * mm),
        sharex=True,                 # <── ⭐ THIS unifies the x-axis
    )

    max_trial_length_all = 0
    discrete_state_matrices_all = [[], []]
    n_trials_all = [0, 0]

    # ── 2.  (unchanged) load every session’s results and accumulate data ────────
    for session_data_name in session_data_names:

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

        fast_model_results_dir, slow_model_results_dir = data_loader.get_model_result_dirs(
            time_offset=time_offset,
            train_test=train_test_option,
            data_format=data_format,
            model_type=model_type,
            dynamics_class=dynamics_class,
            emission_class=emission_class,
            init_type=init_type,
            subspace_type=subspace_type,
            alpha=alpha,
            check_existence=True)

        print(slow_model_results_dir)

        ## Read SLDS processed data
        if model_type in ['LDS']:

            ## Omit discrete states for LDS
            model_save_name = '_'.join(map(str, [x for x in [
                'r' + str(random_state),
                's' + str(n_continuous_states),
                'i' + str(n_iters)]]))
        else:
            model_save_name = '_'.join(map(str, [x for x in [
                'r' + str(random_state),
                's' + str(n_continuous_states),
                'd' + str(n_discrete_states),
                'i' + str(n_iters)]]))

        fast_res_save_path = os.path.join(fast_model_results_dir, model_save_name) + '.pkl'
        slow_res_save_path = os.path.join(slow_model_results_dir, model_save_name) + '.pkl'

        try:
            with open(fast_res_save_path, 'rb') as f:
                res_SLDS_fast = pickle.load(f)
            with open(slow_res_save_path, 'rb') as f:
                res_SLDS_slow = pickle.load(f)
        except:
            print('Model results not found')
            print('Fast:', fast_res_save_path)
            print('Slow:', slow_res_save_path)
            return

        ## Organize data by trial filter
        data_by_trial_filter = {
            trial_filters[0]: {
                'continuous_states': res_SLDS_fast['test_continuous_states'],
                'discrete_states'  : res_SLDS_fast['test_discrete_states'],
                'model'            : res_SLDS_fast['model']
            },
            trial_filters[1]: {
                'continuous_states': res_SLDS_slow['test_continuous_states'],
                'discrete_states'  : res_SLDS_slow['test_discrete_states'],
                'model'            : res_SLDS_slow['model']
            }
        }

        ## Get the max trial length for both slow and fast trials
        for i_tf, trial_filter in enumerate(data_by_trial_filter.keys()):
            data = data_by_trial_filter[trial_filter]
            discrete_states = data['discrete_states']
            trial_lengths = [len(discrete_states_) for discrete_states_ in discrete_states]
            max_trial_length = np.max(trial_lengths)
            max_trial_length_all = max(max_trial_length_all, max_trial_length)

        for i_tf, trial_filter in enumerate(data_by_trial_filter.keys()):
            data = data_by_trial_filter[trial_filter]
            discrete_states = data['discrete_states']
            n_trials = len(discrete_states)
            n_trials_all[i_tf] += n_trials
            trial_lengths = [len(discrete_states_) for discrete_states_ in discrete_states]

            ## Reorder discrete states
            discrete_states, _, _ = utils_vis.reorder_discrete_states(discrete_states, n_discrete_states)

            discrete_state_matrix = np.full((n_trials, max_trial_length_all), -1)

            for i_trial, trial_length in enumerate(trial_lengths):
                discrete_state_matrix[i_trial, :trial_length] = discrete_states[i_trial]

            discrete_state_matrices_all[i_tf].append(discrete_state_matrix)


    # ── 3.  Plot the stacked bars for each trial filter  ────────────────────────
    initial_durations_all = []

    for i_tf, trial_filter in enumerate(data_by_trial_filter.keys()):
        discrete_state_matrix_all = np.full((n_trials_all[i_tf], max_trial_length_all), -1)

        n_trials_cumulative = 0
        for i_session, discrete_state_matrix in enumerate(discrete_state_matrices_all[i_tf]):
            n_trials, max_trial_length = discrete_state_matrix.shape

            discrete_state_matrix_all[n_trials_cumulative:n_trials_cumulative + n_trials, :max_trial_length] = discrete_state_matrix
            n_trials_cumulative += n_trials

        initial_durations = utils_vis.find_initial_zero_durations(discrete_state_matrix_all)
        initial_durations_all.append(initial_durations * time_step)

        # stacked‐bar plot (identical to your original body)  --------------------
        x         = np.arange(max_trial_length_all) * time_step + visual_delay_time
        bar_width = time_step * 0.8
        base      = np.zeros_like(x, dtype=float)

        for i_discrete_state in range(n_discrete_states):
            counts = np.sum(discrete_state_matrix_all == i_discrete_state, axis=0)
            axs[i_tf].bar(
                x, counts,
                bottom=base,
                color=color_palette[trial_filter][i_discrete_state],
                width=bar_width,
                align='edge',
            )
            base += counts

        # y-axis housekeeping (unchanged)  --------------------------------------
        axs[i_tf].set_ylim(0, n_trials_all[i_tf])
        axs[i_tf].set_ylabel(
            trial_filter_name_conversion[trial_filter] + ' trial count',
            fontsize=5,
        )

        axs[i_tf].spines['top'].set_visible(False)  # hide top spine
        axs[i_tf].spines['right'].set_visible(False)  # hide right spine

    # ipdb.set_trace()


    # ── 4.  NOW set the shared x-axis once, for all subplots  ───────────────────
    # ── X-axis ticks and labels (shared) ──────────────────────────────
    xtick_step = 0.5                        # <- or param
    T_plot     = max_trial_length_all
    t_last     = visual_delay_time + (T_plot - 1) * time_step
    
    first_nice = np.ceil(visual_delay_time / xtick_step) * xtick_step
    all_ticks  = np.concatenate([[visual_delay_time],
                                 np.arange(first_nice,
                                           t_last + 1e-9,
                                           xtick_step)])
    tick_times = np.unique(np.round(all_ticks, 10))
    
    axs[1].set_xticks(tick_times)           # absolute coordinates
    axs[1].set_xticklabels(
        [f"{tick_times[0]:.2f}"] + [f"{t:.1f}" for t in tick_times[1:]]
    )
    
    # hide labels on the upper panel
    axs[0].tick_params(axis='x', which='both', labelbottom=False)
    
    # shared limits: start at visual_delay_time
    axs[0].set_xlim(visual_delay_time, t_last)


    # ── 5.  Statistics and figure saving – unchanged ───────────────────────────
    fast  = initial_durations_all[0]
    slow  = initial_durations_all[1]

    # ── classical MW‑U test ───────────────────────────────────────────────────
    stat, p_value = mannwhitneyu(fast, slow, alternative='two-sided')

    # ── descriptive stats ─────────────────────────────────────────────────────
    mean_fast, mean_slow = np.mean(fast), np.mean(slow)

    # ── Rank-biserial r (effect size for MWU) ───────────────────────────────────
    n_fast, n_slow = len(fast), len(slow)
    r_rb = 1 - 2 * stat / (n_fast * n_slow)

    print(f"Fast trial mean:  {mean_fast:.3f}")
    print(f"Slow trial mean:  {mean_slow:.3f}")
    print(f"Mann-Whitney U:   {stat:.0f}   P-value: {p_value:.3g}")
    print(f"Rank-biserial r:  {r_rb:.3f}")

    utils_vis.confidence_interval_95_unpaired(fast, slow)

    plt.tight_layout()

    res_name = '_'.join(map(str, [x for x in [
        task_name,
        'discrete_states_over_time',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        train_test_option,
        random_state,
        n_continuous_states,
        n_discrete_states,
        n_iters,
        model_type,
        dynamics_class,
        emission_class,
        init_type,
        alpha] if x is not None]))

    save_path = os.path.join(vis_dir, res_name + '.pdf')
    plt.savefig(save_path, dpi=600, transparent=True, bbox_inches=None)
    plt.close()


def plot_3D_dynamical_latent_trajectories_integrated(
    session_data_name,
    unit_filter,
    input_unit_filter,
    window_config,
    time_step,
    data_format,
    trial_filters,
    train_test_option,
    random_state,
    n_continuous_states,
    n_discrete_states, 
    n_iters,
    model_type,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    show_individual_trajectories=False,
    show_average_trajectories=True,
    color_by_time_gradient=False,
    color_by_discrete_state=False,
    time_index_marker=None,
    show_turning_points=False,
    show_flow_field=False,
    normalize_flow_field=False,
    show_flow_field_boundary=False,
    show_custom_axes=True,
    view_name='view1',
    visual_delay_time=0,
    supplement_format=False,
    show_preview=False):

    task = session_data_name.split('_')[-1]

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
    data_loader.reformat_firing_rate_data(data_format, trial_length_filter_percentile=90)
    
    fast_model_results_dir, slow_model_results_dir = data_loader.get_model_result_dirs(
        time_offset=time_offset,
        train_test=train_test_option,
        data_format=data_format,
        model_type=model_type,
        dynamics_class=dynamics_class,
        emission_class=emission_class,
        init_type=init_type,
        subspace_type=subspace_type,
        alpha=alpha,
        check_existence=True)
    
    target_ids_fast, target_ids_slow = data_loader.get_target_ids()
    target_ids_fast -= target_ids_fast.min()
    target_ids_slow -= target_ids_slow.min()
    color_palette = target_color_palette_8

    # target_ids = np.concatenate((target_ids_slow, target_ids_fast))

    if data_format == 'resample_avg':
        target_ids_fast = np.arange(8)
        target_ids_slow = np.arange(8)

    if supplement_format:
        fig = plt.figure(figsize=(50*mm, 25*mm))
    else:
        fig = plt.figure(figsize=(90*mm, 45*mm))
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    axs = [ax0, ax1]

    try:
        trajectory_viewing_angles = session_trajectory_viewing_angles[session_data_name][unit_filter][train_test_option][view_name]
    except:
        trajectory_viewing_angles = {'fast': [0, 0, 0], 'slow': [0, 0, 0]}

    if show_custom_axes:
        if supplement_format:
            figsize_coords = (10 * mm, 10 * mm)
        else:
            figsize_coords = (15 * mm, 15 * mm)

        # ── FAST, no labels ───────────────────────────────────────────────────────────
        fig_coords_fast, ax_coords_fast = plt.subplots(
            figsize=figsize_coords,
            subplot_kw={"projection": "3d"}          # ← makes it a 3-D Axes
        )

        # ── FAST, with labels ─────────────────────────────────────────────────────────
        fig_coords_fast_labeled, ax_coords_fast_labeled = plt.subplots(
            figsize=figsize_coords,
            subplot_kw={"projection": "3d"}
        )

        # ── SLOW, no labels ───────────────────────────────────────────────────────────
        fig_coords_slow, ax_coords_slow = plt.subplots(
            figsize=figsize_coords,
            subplot_kw={"projection": "3d"}
        )

        # ── SLOW, with labels ─────────────────────────────────────────────────────────
        fig_coords_slow_labeled, ax_coords_slow_labeled = plt.subplots(
            figsize=figsize_coords,
            subplot_kw={"projection": "3d"}
        )

        # Re-assemble your convenience lists
        figs_coords         = [fig_coords_fast,         fig_coords_slow]
        axs_coords          = [ax_coords_fast,          ax_coords_slow]
        figs_coords_labeled = [fig_coords_fast_labeled, fig_coords_slow_labeled]
        axs_coords_labeled  = [ax_coords_fast_labeled,  ax_coords_slow_labeled]

        # group all four Axes3D objects so we can loop
        coord_axes = [
            ax_coords_fast,
            ax_coords_fast_labeled,
            ax_coords_slow,
            ax_coords_slow_labeled,
        ]

        for ax in coord_axes:
            # 1. hide the grid lines
            ax.grid(False)

            # 2. make the panes transparent and remove their borders
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.fill = False           # no background colour
                axis.pane.set_edgecolor('none')  # no outline
                axis.line.set_visible(False)      # no axis line

            # (optional) strip ticks as well
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

    if supplement_format:
        cbar_time_figsize = (20*mm, 5*mm)
    else:
        cbar_time_figsize = (30*mm, 10*mm)
    fig_cbar_time, ax_cbar_time = plt.subplots(figsize=cbar_time_figsize)
    fig_cbar_flow_field, axs_cbar_flow_field = plt.subplots(n_discrete_states, 1, figsize=(30*mm, 5*mm*n_discrete_states))

    ## Read SLDS processed data
    if model_type in ['LDS']:

        ## Omit discrete states for LDS
        model_save_name = '_'.join(map(str, [x for x in [
            'r' + str(random_state),
            's' + str(n_continuous_states),
            'i' + str(n_iters)]]))
    else:
        model_save_name = '_'.join(map(str, [x for x in [
            'r' + str(random_state),
            's' + str(n_continuous_states),
            'd' + str(n_discrete_states),
            'i' + str(n_iters)]]))

    fast_res_save_path = os.path.join(fast_model_results_dir, model_save_name) + '.pkl'
    slow_res_save_path = os.path.join(slow_model_results_dir, model_save_name) + '.pkl'

    try:
        with open(fast_res_save_path, 'rb') as f:
            res_SLDS_fast = pickle.load(f)
        with open(slow_res_save_path, 'rb') as f:
            res_SLDS_slow = pickle.load(f)
    except:
        print('Model results not found')
        return
    
    ## Organize data by trial filter
    data_by_trial_filter = {
        trial_filters[0]: {
            'target_ids'       : target_ids_fast,
            'continuous_states': res_SLDS_fast['test_continuous_states'],
            'discrete_states'  : res_SLDS_fast['test_discrete_states'],
            'model'            : res_SLDS_fast['model']},
        trial_filters[1]: {
            'target_ids'       : target_ids_slow,
            'continuous_states': res_SLDS_slow['test_continuous_states'],
            'discrete_states'  : res_SLDS_slow['test_discrete_states'],
            'model'            : res_SLDS_slow['model']} 
    }

    ## Set colors or cmaps depending on if the flow fields are to be normalized
    if normalize_flow_field:
        colors_or_cmaps = discrete_state_quiver_cmaps
    else:
        colors_or_cmaps = discrete_state_colors

    ## Keep track of the min and max values for the continuous states to set the axis limits
    # x_min = np.inf
    # x_max = -np.inf
    # y_min = np.inf
    # y_max = -np.inf
    # z_min = np.inf
    # z_max = -np.inf

    trial_length_fast_max = np.max([continuous_states_fast.shape[0] for continuous_states_fast in data_by_trial_filter[trial_filters[0]]['continuous_states']])
    trial_length_slow_max = np.max([continuous_states_slow.shape[0] for continuous_states_slow in data_by_trial_filter[trial_filters[1]]['continuous_states']])
    trial_length_max = max(trial_length_fast_max, trial_length_slow_max)

    for i_tf, data in enumerate(data_by_trial_filter.values()):

        target_ids        = data['target_ids']
        continuous_states = data['continuous_states']
        discrete_states   = data['discrete_states']
        model             = data['model']

        ## Reorder discrete states
        discrete_states, _, reorder = utils_vis.reorder_discrete_states(discrete_states, n_discrete_states)

        if n_continuous_states > 3:
            continuous_states = utils_vis.reduce_dynamical_dimensionality( 
                model, 
                continuous_states,
                n_target_dimensions=3)
            
            # continuous_states = utils_vis.reduce_dynamical_dimensionality_( 
            #     model, 
            #     continuous_states,
            #     n_target_dimensions=3)
            
            # continuous_states = utils_vis.reduce_dynamical_dimensionality_by_fitting( 
            #     model, 
            #     continuous_states,
            #     discrete_states,
            #     n_discrete_states,
            #     n_target_dimensions=3)

        # x_min = min(x_min, utils_processing.min_list_of_2d_np(continuous_states, last_dim=0))
        # x_max = max(x_max, utils_processing.max_list_of_2d_np(continuous_states, last_dim=0))
        # y_min = min(y_min, utils_processing.min_list_of_2d_np(continuous_states, last_dim=1))
        # y_max = max(y_max, utils_processing.max_list_of_2d_np(continuous_states, last_dim=1))
        # z_min = min(z_min, utils_processing.min_list_of_2d_np(continuous_states, last_dim=2))
        # z_max = max(z_max, utils_processing.max_list_of_2d_np(continuous_states, last_dim=2))

        x_min = utils_processing.min_list_of_2d_np(continuous_states, last_dim=0)
        x_max = utils_processing.max_list_of_2d_np(continuous_states, last_dim=0)
        y_min = utils_processing.min_list_of_2d_np(continuous_states, last_dim=1)
        y_max = utils_processing.max_list_of_2d_np(continuous_states, last_dim=1)
        z_min = utils_processing.min_list_of_2d_np(continuous_states, last_dim=2)
        z_max = utils_processing.max_list_of_2d_np(continuous_states, last_dim=2)

        ## Plot trajectories for all trials
        if show_individual_trajectories:

            if color_by_time_gradient:
                for i_cs, continuous_states_ in enumerate(continuous_states):
                    utils_vis.add_color_graded_trajectory_3D(
                        axs[i_tf], 
                        continuous_states_,
                        time_gradient_cmap,
                        linewidth=0.25,
                        alpha=0.2,
                        t=np.arange(continuous_states_.shape[0]) * time_step,
                        t_max=trial_length_max * time_step)
            
            elif color_by_discrete_state:
                cmap = color_maps[task][trial_filters[i_tf]]

                for i_cs, (continuous_states_, discrete_states_) in enumerate(zip(continuous_states, discrete_states)):
                    utils_vis.add_color_graded_trajectory_3D(
                        axs[i_tf], 
                        continuous_states_,
                        cmap,
                        linewidth=0.25,
                        alpha=0.9,
                        t=discrete_states_,
                        t_max=n_discrete_states - 1)
            else:
                for i_cs, continuous_states_ in enumerate(continuous_states):
                    target_id = target_ids[i_cs]
                    axs[i_tf].plot(
                        continuous_states_[:, 0], 
                        continuous_states_[:, 1], 
                        continuous_states_[:, 2],
                        color=color_palette[target_id - 1],
                        lw=0.25,
                        alpha=0.2)
            
            if not show_average_trajectories and isinstance(time_index_marker, int):
                for i_cs, continuous_states_ in enumerate(continuous_states):
                    axs[i_tf].scatter(
                        continuous_states_[time_index_marker, 0],
                        continuous_states_[time_index_marker, 1],
                        continuous_states_[time_index_marker, 2],
                        # color=time_gradient_cmap(time_index_marker / trial_length_max),
                        color=caltech_orange,
                        # color='#D62800',
                        s=50,
                        alpha=alpha_line)
                    
                    # if i_tf == 0:
                    #     axs[i_tf].scatter(
                    #         continuous_states_[trial_length_fast_max, 0],
                    #         continuous_states_[trial_length_fast_max, 1],
                    #         continuous_states_[trial_length_fast_max, 2],
                    #         # color=caltech_orange,
                    #         color='magenta',
                    #         s=50,
                    #         alpha=alpha_line_thin)
            
            if not show_average_trajectories and show_turning_points:
                for i_cs, continuous_states_ in enumerate(continuous_states):

                    turning_point_indices, trajectory_angles = utils_vis.extract_turning_points(
                        continuous_states_, 
                        index_min=8,
                        index_max=50)

                    # print('i_cs:', i_cs, ' ', turning_point_indices[-1])
                    # print(turning_point_indices)
                    # print(trajectory_angles)
                    # print(turning_point_indices[-1])

                    chosen_turning_point_index = turning_point_indices[-1]

                    axs[i_tf].scatter(
                        continuous_states_[chosen_turning_point_index, 0],
                        continuous_states_[chosen_turning_point_index, 1],
                        continuous_states_[chosen_turning_point_index, 2],
                        # color=time_gradient_cmap(time_index_marker / trial_length_max),
                        color=caltech_orange,
                        # color='#D62800',
                        s=50,
                        alpha=alpha_line)                     
                
        ## Plot trajectories averaged by target
        target_ids_unique = np.unique(target_ids)
        if show_average_trajectories:

            ## Construct time data
            trial_lengths = [len(discrete_states_) for discrete_states_ in discrete_states]
            times = [np.arange(trial_length) * time_step for trial_length in trial_lengths]

            ## Resample continuous states, discrete states, and times to have the same length
            if data_format is None:
                continuous_states_aligned, _, _ = utils_processing.resample_emissions(continuous_states)
                discrete_states_aligned         = utils_processing.resample_discrete_states(discrete_states, trial_lengths)
                times_aligned                   = utils_processing.resample_times(times, trial_lengths)

                continuous_states_aligned = np.array(continuous_states_aligned)
                discrete_states_aligned   = np.array(discrete_states_aligned)
                times_aligned             = np.array(times_aligned)
            else:
                continuous_states_aligned = np.array(continuous_states)
                discrete_states_aligned   = np.array(discrete_states)
                times_aligned             = np.array(times)

            for target_id in target_ids_unique:
                # ipdb.set_trace()
                target_filter = (target_ids == target_id)
                continuous_states_avg = np.mean(continuous_states_aligned[target_filter, :, :], axis=0)
                discrete_states_avg   = np.mean(discrete_states_aligned[target_filter],         axis=0)
                times_avg             = np.mean(times_aligned[target_filter],                   axis=0)

                if color_by_time_gradient:
                    utils_vis.add_color_graded_trajectory_3D(
                        axs[i_tf], 
                        continuous_states_avg,
                        time_gradient_cmap,
                        linewidth=size_line_thin,
                        alpha=1,
                        t=times_avg,
                        t_max=trial_length_max * time_step)
                    
                elif color_by_discrete_state:
                    # if reorder:
                    #     cmap = color_maps['gray'][trial_filters[i_tf]]
                    # else:
                    #     cmap = color_maps['gray_r'][trial_filters[i_tf]]

                    # cmap = color_maps[task][trial_filters[i_tf]]

                    cmap = color_maps['gray_r'][trial_filters[i_tf]]

                    utils_vis.add_color_graded_trajectory_3D(
                        axs[i_tf], 
                        continuous_states_avg,
                        cmap,
                        linewidth=size_line_thin,
                        alpha=alpha_line_thick,
                        t=discrete_states_avg,
                        t_max=n_discrete_states - 1)

                else:
                    axs[i_tf].plot(
                        continuous_states_avg[:, 0], 
                        continuous_states_avg[:, 1], 
                        continuous_states_avg[:, 2],
                        color=color_palette[target_id - 1],
                        lw=size_line_thin,
                        alpha=alpha_line)

                if isinstance(time_index_marker, int): 
                    axs[i_tf].scatter(
                        continuous_states_avg[time_index_marker, 0],
                        continuous_states_avg[time_index_marker, 1],
                        continuous_states_avg[time_index_marker, 2],
                        color=caltech_orange,
                        s=50,
                        alpha=alpha_point)     

                if show_turning_points:

                    turning_point_indices, trajectory_angles = utils_vis.extract_turning_points(
                        continuous_states_avg, 
                        index_min=8,
                        index_max=50)

                    # print('i_cs:', i_cs, ' ', turning_point_indices[-1])
                    # print(turning_point_indices)
                    # print(trajectory_angles)
                    # print(turning_point_indices[-1])

                    # axs[i_tf].scatter(
                    #     continuous_states_avg[turning_point_indices[-1], 0],
                    #     continuous_states_avg[turning_point_indices[-1], 1],
                    #     continuous_states_avg[turning_point_indices[-1], 2],
                    #     # color=time_gradient_cmap(time_index_marker / trial_length_max),
                    #     color=caltech_orange,
                    #     # color='#D62800',
                    #     s=100,
                    #     alpha=alpha_line)      

                    axs[i_tf].plot(
                        continuous_states_avg[turning_point_indices[-1] - 1 : turning_point_indices[-1] + 1, 0],
                        continuous_states_avg[turning_point_indices[-1] - 1 : turning_point_indices[-1] + 1, 1],
                        continuous_states_avg[turning_point_indices[-1] - 1 : turning_point_indices[-1] + 1, 2],
                        # color=time_gradient_cmap(time_index_marker / trial_length_max),
                        color=caltech_orange,
                        # color='#D62800',
                        # s=100,
                        alpha=alpha_line)      
                    
                    # axs[i_tf].scatter(
                    #     continuous_states_avg[turning_point_indices[0], 0],
                    #     continuous_states_avg[turning_point_indices[0], 1],
                    #     continuous_states_avg[turning_point_indices[0], 2],
                    #     # color=time_gradient_cmap(time_index_marker / trial_length_max),
                    #     color=caltech_orange,
                    #     # color='#D62800',
                    #     s=50,
                    #     alpha=alpha_line)  


        if show_custom_axes:
            utils_vis.add_axes(
                axs_coords[i_tf],
                x_min * 0.5, x_max * 0.5,
                y_min * 0.5, y_max * 0.5,
                z_min * 0.5, z_max * 0.5,
                lw=0.3,
                alpha=1,
                label_fontsize=5,
                show_labels=False,
                headsize=3 if supplement_format else 5)
            
            utils_vis.add_axes(
                axs_coords_labeled[i_tf],
                x_min * 0.5, x_max * 0.5,
                y_min * 0.5, y_max * 0.5,
                z_min * 0.5, z_max * 0.5,
                lw=0.3,
                alpha=1,
                label_fontsize=5,
                show_labels=True,
                headsize=3 if supplement_format else 5)

            axs_coords[i_tf].view_init(elev=trajectory_viewing_angles[trial_filters[i_tf]][0], azim=trajectory_viewing_angles[trial_filters[i_tf]][1], roll=trajectory_viewing_angles[trial_filters[i_tf]][2])
            axs_coords_labeled[i_tf].view_init(elev=trajectory_viewing_angles[trial_filters[i_tf]][0], azim=trajectory_viewing_angles[trial_filters[i_tf]][1], roll=trajectory_viewing_angles[trial_filters[i_tf]][2])


    if color_by_time_gradient:

        # ── normalisation & colormap ───────────────────────────────────────
        t_start = visual_delay_time
        t_end   = visual_delay_time + trial_length_max * time_step   # inclusive RHS

        norm = plt.Normalize(t_start, t_end)
        cmap = time_gradient_cmap
        sm   = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        # ── build the tick vector exactly like for the x-axis ──────────────
        if supplement_format:
            xtick_step = 2.0
        else:
            xtick_step = 1.0
        first_nice = np.ceil(t_start / xtick_step) * xtick_step
        all_ticks  = np.concatenate((
            [t_start],
            np.arange(first_nice, t_end + 1e-9, xtick_step)
        ))
        ticks = np.unique(np.round(all_ticks, 10))   # kill FP noise

        # ── create the horizontal colour-bar ───────────────────────────────
        cbar = fig_cbar_time.colorbar(
            sm,
            cax=ax_cbar_time,
            orientation='horizontal'
        )
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(
            [f"{ticks[0]:.2f}"] + [f"{t:.1f}" for t in ticks[1:]]
        )
        # cbar.set_label('Time (s)', fontsize=5)
        cbar.ax.tick_params(labelsize=5, length=0, width=0)

        # ── tidy up ────────────────────────────────────────────────────────
        cbar.outline.set_visible(False)
        for spine in ax_cbar_time.spines.values():
            spine.set_visible(False)
        
        if supplement_format:
            fig_cbar_time.subplots_adjust(left=0.1, right=0.9, top=0.75, bottom=0.60)
        else:
            fig_cbar_time.subplots_adjust(left=0.03, right=0.97, top=0.75, bottom=0.60)
            
        cbar.ax.xaxis.set_label_position("top")     # label above
        cbar.ax.xaxis.set_ticks_position("bottom")  # ticks below


    for i_tf, data in enumerate(data_by_trial_filter.values()):

        target_ids        = data['target_ids']
        continuous_states = data['continuous_states']
        discrete_states   = data['discrete_states']
        model             = data['model']

        ## Reorder discrete states
        _, _, reorder = utils_vis.reorder_discrete_states(discrete_states, n_discrete_states)

        x_min = utils_processing.min_list_of_2d_np(continuous_states, last_dim=0)
        x_max = utils_processing.max_list_of_2d_np(continuous_states, last_dim=0)
        y_min = utils_processing.min_list_of_2d_np(continuous_states, last_dim=1)
        y_max = utils_processing.max_list_of_2d_np(continuous_states, last_dim=1)
        z_min = utils_processing.min_list_of_2d_np(continuous_states, last_dim=2)
        z_max = utils_processing.max_list_of_2d_np(continuous_states, last_dim=2)

        vmax_vals = [4, 4]

        if show_flow_field:
            if model_type == 'roSLDS':
                quivers, color_norms = utils_vis.add_SLDS_dynamics_quiver_3D(
                    axs[i_tf], 
                    model,
                    colors_or_cmaps,
                    n_points=6,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    z_min=z_min,
                    z_max=z_max,
                    length=10,
                    normalize=normalize_flow_field)

                for k, (quiv, norm) in enumerate(zip(quivers, color_norms)):
                    if norm is not None:  # Only add a colorbar if normalization was applied
                        sm = cm.ScalarMappable(norm=norm, cmap=colors_or_cmaps[k])
                        sm.set_array([])
                        cbar = plt.colorbar(
                            sm, 
                            cax=axs_cbar_flow_field[i_tf, k],
                            orientation='vertical')
                        if k == 0:
                            state_name = 'Transient Phase'
                        elif k == 1:
                            state_name = 'Steady Phase'
                        cbar.set_label(f'{state_name} Magnitude (a.u.)', fontsize=label_fontsize)


            elif model_type in ['rSLDS', 'SLDS']:

                ## Resample continuous states to have the same length
                if data_format is None:
                    continuous_states_aligned, _, _ = utils_processing.resample_emissions(continuous_states)
                    discrete_states_aligned = utils_processing.resample_discrete_states(discrete_states)

                    continuous_states_aligned = np.array(continuous_states_aligned)
                    discrete_states_aligned   = np.array(discrete_states_aligned)
                else:
                    continuous_states_aligned = np.array(continuous_states)
                    discrete_states_aligned   = np.array(discrete_states)

                ## Average continuous and discrete states by target
                continuous_state_target_avgs = []
                discrete_state_target_avgs   = []

                for target_id in target_ids_unique:
                    target_filter = (target_ids == target_id)
                    continuous_states_avg = np.mean(continuous_states_aligned[target_filter, :, :], axis=0)
                    discrete_states_avg   = np.mean(discrete_states_aligned[target_filter], axis=0)

                    continuous_state_target_avgs.append(continuous_states_avg)
                    discrete_state_target_avgs.append(discrete_states_avg)

                continuous_state_target_avgs = np.array(continuous_state_target_avgs)
                discrete_state_target_avgs   = np.round(discrete_state_target_avgs).astype(np.int32)

                # quivers, color_norms = utils_vis.visualize_SLDS_flow_fields_occupancy_grid(
                #     axs[i_tf],
                #     model,
                #     colors_or_cmaps,
                #     continuous_state_target_avgs,
                #     discrete_state_target_avgs,
                #     n_discrete_states,
                #     n_points=10,
                #     length=10,
                #     alpha=0.3,
                #     normalize=normalize_flow_field)
                
                quivers, color_norms = utils_vis.visualize_SLDS_flow_fields(
                    axs[i_tf],
                    model,
                    colors_or_cmaps,
                    continuous_state_target_avgs,
                    discrete_state_target_avgs,
                    n_discrete_states,
                    vmin=[0, 0],
                    vmax=vmax_vals,
                    index_spacing=2,
                    length=8,
                    alpha=alpha_line,
                    normalize=normalize_flow_field,
                    reverse=reorder,
                    head_ratio=0.5 if supplement_format else 0.2)

                ## Add colorbars for each quiver
                if i_tf == 0:
                    # for k, (quiv, norm) in enumerate(zip(quivers, color_norms)):
                    #     if norm is not None:  # Only add a colorbar if normalization was applied
                    #         sm = cm.ScalarMappable(norm=norm, cmap=colors_or_cmaps[k])
                    #         sm.set_array([])
                    #         cbar = plt.colorbar(
                    #             sm, 
                    #             cax=axs_cbar_flow_field[k],
                    #             orientation='horizontal')
                            # if k == 0:
                            #     state_name = 'Transient Phase'
                            # elif k == 1:
                            #     state_name = 'Steady Phase'
                            # cbar.set_label(f'{state_name} Magnitude (a.u.)', fontsize=5)

                    # pick something sensible once, e.g. 5 ticks between 0‒1
                    # tick_step = 0.2
                    # ticks_shared = np.round(np.arange(0, 1 + 1e-9, tick_step), 2)
                    # ticks_shared = [0, vmax_vals[i_tf]]

                    bar_frac = 0.35          # → keep 35 % of the original height

                    for k, (quiv, norm) in enumerate(zip(quivers, color_norms)):
                        if norm is None:          # skip quiver-groups that have no norm
                            continue
                        
                        sm = cm.ScalarMappable(norm=norm, cmap=colors_or_cmaps[k])
                        sm.set_array([])

                        # ------------------------------------------------------------------
                        # make the colour-bar
                        # ------------------------------------------------------------------
                        cbar = plt.colorbar(
                            sm,
                            cax=axs_cbar_flow_field[k],          # pre-created axes
                            orientation='horizontal'
                        )

                        # ── slim it down ──────────────────────────────────────────────────
                        pos = cbar.ax.get_position()             # [x0, y0, w, h] in fig-coords
                        new_h = pos.height * bar_frac
                        cbar.ax.set_position([
                            pos.x0,
                            pos.y0 + (pos.height - new_h) / 2,   # vertically centre the strip
                            pos.width,
                            new_h
                        ])
                        cbar.ax.set_aspect('auto')               # keep aspect free

                        # ── ticks: only show them on the *bottom* bar ─────────────────────
                        # cbar.set_ticks(ticks_shared)
                        cbar.set_ticks([])  # no ticks

                        if k < len(quivers) - 1:                 # every bar except last
                            cbar.ax.tick_params(
                                bottom=False,                    # hide tick marks
                                labelbottom=False                # hide labels
                            )
                        else:                                    # last (bottom) bar
                            cbar.ax.tick_params(
                                labelsize=5,
                                bottom=True, length=2, width=0.5
                            )

                        # ── cosmetic clean-up ─────────────────────────────────────────────
                        cbar.outline.set_visible(False)
                        for spine in cbar.ax.spines.values():
                            spine.set_visible(False)


            else:
                quiver, norm = utils_vis.add_LDS_dynamics_quiver_3D(
                    axs[i_tf], 
                    model,
                    colors_or_cmaps[0],
                    n_points=6,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    z_min=z_min,
                    z_max=z_max,
                    length=10,
                    normalize=normalize_flow_field)
                
                if norm is not None:  # Only add a colorbar if normalization was applied
                    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colors_or_cmaps[0]), ax=axs[i_tf])
                    cbar.set_label(f'Magnitude (a.u.)')
                    cbar.set_ticks([])
                    cbar.set_ticklabels([])
        
        if show_flow_field_boundary and model_type == 'roSLDS':
            ## Visualize the state boundaries
                utils_vis.plot_state_boundaries(
                    axs[i_tf], 
                    model,
                    n_points=10,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    z_min=z_min,
                    z_max=z_max)

        ## Set axis limits
        axs[i_tf].set_xlim(x_min, x_max)
        axs[i_tf].set_ylim(y_min, y_max)
        axs[i_tf].set_zlim(z_min, z_max)

        ## Remove background panes
        axs[i_tf].xaxis.pane.fill = False
        axs[i_tf].yaxis.pane.fill = False
        axs[i_tf].zaxis.pane.fill = False

        # Remove the gridlines
        axs[i_tf].grid(False)

        ## Remove axis ticks and labels
        axs[i_tf].set_xticks([])
        axs[i_tf].set_yticks([])
        axs[i_tf].set_zticks([])

        axs[i_tf].set_xticklabels([])
        axs[i_tf].set_yticklabels([])
        axs[i_tf].set_zticklabels([])

        ## Set grid colors
        axs[i_tf].xaxis._axinfo['grid'].update(color='black')
        axs[i_tf].yaxis._axinfo['grid'].update(color='black')
        axs[i_tf].zaxis._axinfo['grid'].update(color='black')

        ## Make pane background transparent
        axs[i_tf].xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  
        axs[i_tf].yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axs[i_tf].zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        if show_custom_axes:
            axs[i_tf].xaxis.line.set_visible(False)
            axs[i_tf].yaxis.line.set_visible(False)
            axs[i_tf].zaxis.line.set_visible(False)
        else:
            ## Set axis labels
            axs[i_tf].set_xlabel('Latent Dimension 1', fontsize=label_fontsize)
            axs[i_tf].set_ylabel('Latent Dimension 2', fontsize=label_fontsize)
            axs[i_tf].set_zlabel('Latent Dimension 3', fontsize=label_fontsize)

    axs[0].view_init(elev=trajectory_viewing_angles[trial_filters[0]][0], azim=trajectory_viewing_angles[trial_filters[0]][1], roll=trajectory_viewing_angles[trial_filters[0]][2])
    axs[1].view_init(elev=trajectory_viewing_angles[trial_filters[1]][0], azim=trajectory_viewing_angles[trial_filters[1]][1], roll=trajectory_viewing_angles[trial_filters[1]][2])


    ## Set colorbar
    # cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
    # sm.set_array([])
    # plt.colorbar(sm, cax=cbar_ax, label='Arrow Magnitude')

    ## Animation function: this is called sequentially
    # def update(frame):
    #     elev = 30  # fixed elevation
    #     azim = frame  # rotate 360 degrees around the Z-axis
    #     ax0.view_init(elev=elev, azim=azim)
    #     ax1.view_init(elev=elev, azim=azim)

    #     return fig

    # ani = FuncAnimation(fig, update, frames=np.arange(0, 360), interval=50)

    ## Set big title
    # fig.suptitle(session_data_name)

    # plt.tight_layout()

    fig.subplots_adjust(
        left=0.01, right=0.99,                    # outer frame
        top=0.99, bottom=0.01,
        wspace=0                               # gap between the 3-D panels
    )

    ## Write image
    # session_vis_dir = os.path.join(vis_dir, session_data_name)

    # if not os.path.isdir(session_vis_dir):
    #     os.makedirs(session_vis_dir)

    res_name = '_'.join(map(str, [x for x in [
        session_data_name,
        'dynamical_latent_trajectories_3d',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        n_continuous_states,
        n_discrete_states,
        n_iters,
        model_type,
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha,
        ] if x is not None]))
    
    if show_individual_trajectories:
        res_name += '_it'
    if show_average_trajectories:
        res_name += '_at'
    if color_by_time_gradient:
        res_name += '_ctg'
    if color_by_discrete_state:
        res_name += '_cds'
    if time_index_marker is not None:
        res_name += '_tim' + str(time_index_marker)
    if show_flow_field:
        res_name += '_ff'
    if normalize_flow_field:
        res_name += '_normalized'
    if show_flow_field_boundary:
        res_name += '_ffb'
    if show_custom_axes:
        res_name += '_ca'
    res_name += '_' + str(view_name)
    
    save_path = os.path.join(vis_dir, res_name + '.pdf')

    fig.savefig(save_path, dpi=600, transparent=True, bbox_inches=None)
    if color_by_time_gradient:
        fig_cbar_time.tight_layout()
        fig_cbar_time.savefig(save_path.replace('.pdf', '_cbar_time.pdf'), dpi=600, transparent=True, bbox_inches=None)
    if normalize_flow_field:
        fig_cbar_flow_field.tight_layout()
        fig_cbar_flow_field.savefig(save_path.replace('.pdf', '_cbar_flow_field.pdf'), dpi=600, transparent=True, bbox_inches=None)
    if show_custom_axes:
        figs_coords[0].savefig(save_path.replace('.pdf', '_coords_fast.pdf'), dpi=600, transparent=True, bbox_inches=None)
        figs_coords[1].savefig(save_path.replace('.pdf', '_coords_slow.pdf'), dpi=600, transparent=True, bbox_inches=None)
        figs_coords_labeled[0].savefig(save_path.replace('.pdf', '_coords_labeled_fast.pdf'), dpi=600, transparent=True, bbox_inches=None)
        figs_coords_labeled[1].savefig(save_path.replace('.pdf', '_coords_labeled_slow.pdf'), dpi=600, transparent=True, bbox_inches=None)

    if show_preview:
        plt.show()

    print('(Axis 0) Elevation angle:', axs[0].elev, ' Azimuth angle:', axs[0].azim)
    print('(Axis 1) Elevation angle:', axs[1].elev, ' Azimuth angle:', axs[1].azim)
    plt.close(fig)
    plt.close(fig_cbar_time)
    plt.close(fig_cbar_flow_field)
    plt.close(figs_coords[0])
    plt.close(figs_coords[1])
    plt.close(figs_coords_labeled[0])
    plt.close(figs_coords_labeled[1])


def plot_dynamical_latent_trajectories_per_dimension(
    session_data_names,
    unit_filter,
    input_unit_filter,
    window_config,
    time_step,
    data_format,
    trial_filters,
    train_test_option,
    random_state,
    n_continuous_states,
    n_discrete_states, 
    n_iters,
    model_type,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    visual_delay_time,
    peak_time,
    significance_alpha: float = 0.05,
    correction_method: str = "fdr_bh",
    save_significance_data=False):

    data_by_trial_filter_all = {
        trial_filters[0]: {'continuous_states_by_target' : [[] for _ in range(8)]},
        trial_filters[1]: {'continuous_states_by_target' : [[] for _ in range(8)]}}

    subject_task_str = session_data_names[0].split('_')[0][-2:] + '_' + session_data_names[0].split('_')[-1]

    res_name = '_'.join(map(str, [x for x in [
        'dynamical_latent_trajectories_per_dimension',
        subject_task_str,
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        n_continuous_states,
        n_discrete_states,
        n_iters,
        model_type,
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))
    
    res_name += '_vdt' + str(visual_delay_time)
    res_name += '_pt' + str(peak_time)

    visual_delay_index = int(visual_delay_time / time_step)
    peak_index = int((peak_time - visual_delay_time) / time_step)

    ## How many sig. samples does each latent dimension have?
    ## Only count samples that are significant after the peak time and not before
    n_targets = 8
    sig_counts_before_peak = np.zeros((len(session_data_names), n_continuous_states, n_targets), dtype=int)
    sig_counts_after_peak  = np.zeros((len(session_data_names), n_continuous_states, n_targets), dtype=int)

    for i_session, session_data_name in enumerate(session_data_names):

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
        data_loader.reformat_firing_rate_data(data_format, trial_length_filter_percentile=90)

        fast_model_results_dir, slow_model_results_dir = data_loader.get_model_result_dirs(
            time_offset=time_offset,
            train_test=train_test_option,
            data_format=data_format,
            model_type=model_type,
            dynamics_class=dynamics_class,
            emission_class=emission_class,
            init_type=init_type,
            subspace_type=subspace_type,
            alpha=alpha,
            check_existence=True)

        target_ids_fast, target_ids_slow = data_loader.get_target_ids()
        target_ids_fast -= target_ids_fast.min()
        target_ids_slow -= target_ids_slow.min()
        color_palette = target_color_palette_8

        # target_ids = np.concatenate((target_ids_slow, target_ids_fast))

        if data_format == 'resample_avg':
            target_ids_slow = np.arange(8)
            target_ids_fast = np.arange(8)

        # fig, axs = plt.subplots(n_continuous_states, 8, figsize=(90*mm, 10*mm*n_continuous_states), sharex=True, sharey=True)

        fig, axs = plt.subplots(
            n_continuous_states, 8,
            figsize=(90 * mm, 90 * mm * n_continuous_states / 8),  # width / height = cols / rows
            sharex=True, sharey=True,
            constrained_layout=True,   # nicer than tight_layout when using box-aspect
        )

        for ax in axs.flat:            # make each little panel a square
            ax.set_box_aspect(1)       # 1 → width == height

        ## Read SLDS processed data
        if model_type in ['LDS']:

            ## Omit discrete states for LDS
            model_save_name = '_'.join(map(str, [x for x in [
                'r' + str(random_state),
                's' + str(n_continuous_states),
                'i' + str(n_iters)]]))
        else:
            model_save_name = '_'.join(map(str, [x for x in [
                'r' + str(random_state),
                's' + str(n_continuous_states),
                'd' + str(n_discrete_states),
                'i' + str(n_iters)]]))

        slow_res_save_path = os.path.join(slow_model_results_dir, model_save_name) + '.pkl'
        fast_res_save_path = os.path.join(fast_model_results_dir, model_save_name) + '.pkl'

        try:
            with open(slow_res_save_path, 'rb') as f:
                res_SLDS_slow = pickle.load(f)

            with open(fast_res_save_path, 'rb') as f:
                res_SLDS_fast = pickle.load(f)
        except:
            print('Model results not found')
            return

        ## Organize data by trial filter
        data_by_trial_filter = {
            trial_filters[0]: {
                'target_ids'       : target_ids_fast,
                'continuous_states': res_SLDS_fast['test_continuous_states'],
                'discrete_states'  : res_SLDS_fast['test_discrete_states'],
                'model'            : res_SLDS_fast['model'],
                'linestyle'        : '--',
            },
            trial_filters[1]: {
                'target_ids'       : target_ids_slow,
                'continuous_states': res_SLDS_slow['test_continuous_states'],
                'discrete_states'  : res_SLDS_slow['test_discrete_states'],
                'model'            : res_SLDS_slow['model'],
                'linestyle'        : '-',
            }
        }

        for trial_filter, data in data_by_trial_filter.items():

            target_ids        = data['target_ids']
            continuous_states = data['continuous_states']
            discrete_states   = data['discrete_states']
            linestyle         = data['linestyle']

            ## Plot trajectories averaged by target
            target_ids_unique = np.unique(target_ids)

            ## Construct time data
            trial_lengths = [len(discrete_states_) for discrete_states_ in discrete_states]
            times = [np.arange(trial_length) * time_step for trial_length in trial_lengths]

            ## Resample continuous states, discrete states, and times to have the same length
            if data_format is None:
                continuous_states_aligned, _, _ = utils_processing.truncate_emissions(continuous_states)
                # discrete_states_aligned         = utils_processing.resample_discrete_states(discrete_states, trial_lengths)
                # times_aligned                   = utils_processing.resample_times(times, trial_lengths)

                continuous_states_aligned = np.array(continuous_states_aligned)
                # discrete_states_aligned   = np.array(discrete_states_aligned)
                # times_aligned             = np.array(times_aligned)
            else:
                continuous_states_aligned = np.array(continuous_states)
                # discrete_states_aligned   = np.array(discrete_states)
                # times_aligned             = np.array(times)

            for i_target, target_id in enumerate(target_ids_unique):
                target_filter = (target_ids == target_id)
                continuous_states_avg = np.mean(continuous_states_aligned[target_filter, :, :], axis=0)
                continuous_states_sem = np.std(continuous_states_aligned[target_filter, :, :], axis=0) / np.sqrt(np.sum(target_filter))

                data_by_trial_filter_all[trial_filter]['continuous_states_by_target'][i_target] = continuous_states_aligned[target_filter, :, :]

                for i_cs in range(n_continuous_states):
                    axs[i_cs, i_target].plot(
                        continuous_states_avg[:, i_cs],
                        color=color_palette[target_id - 1],
                        lw=0.25,
                        alpha=alpha_line,
                        linestyle=linestyle)

                    axs[i_cs, i_target].fill_between(
                        np.arange(continuous_states_avg.shape[0]),
                        continuous_states_avg[:, i_cs] - continuous_states_sem[:, i_cs],
                        continuous_states_avg[:, i_cs] + continuous_states_sem[:, i_cs],
                        color=color_palette[target_id - 1],
                        alpha=alpha_line_thin,
                        linewidth=0)

                    axs[i_cs, i_target].spines['top'].set_visible(False)
                    axs[i_cs, i_target].spines['right'].set_visible(False)
                    axs[i_cs, i_target].axvline(peak_index, color='k', linestyle='--', linewidth=0.25, alpha=0.5)
    

        # ... draw all panels (do NOT touch ticks inside the loop) ...

        label_ax = axs[n_continuous_states - 1, 0]           # lower-left keeps the labels
        for r in range(n_continuous_states):
            for c in range(8):
                ax = axs[r, c]
                if ax is label_ax or not ax.get_visible():
                    continue               # leave the master axis alone
                ax.tick_params(axis="both", which="both",
                               labelbottom=False, labelleft=False,
                               bottom=False, left=False)   # hides only on this axes

        # after the big plotting loop, once per figure
        T_max = max(tgt_cs.shape[1] for tgt_cs in data_by_trial_filter_all[trial_filters[0]]['continuous_states_by_target'])
        label_ax.set_xlim(0, T_max - 1)             # only touch the “master” axis

        ## Set x axis
        xtick_step = 0.5

        # ───────────── 1. build the tick-time list ─────────────
        t_last      = visual_delay_time + (T_max - 1) * time_step
        first_nice  = (np.ceil(visual_delay_time / xtick_step)) * xtick_step      # 0.150 in the example
        tick_times  = np.concatenate([[visual_delay_time],
                                      np.arange(first_nice, t_last + 1e-9, xtick_step)])
        # Guard against a duplicate when t0 is itself “nice”
        tick_times  = np.unique(np.round(tick_times, 10))

        # ───────────── 2. convert times → sample indices ───────
        tick_idx = (tick_times - visual_delay_time) / time_step     # index 0 for t0, >0 for the rest

        # ───────────── 3. apply to every axis you care about ───
        axs[0, 0].set_xticks(tick_idx)
        axs[0, 0].set_xticklabels([f'{tick_times[0]:.2f}'] + [f'{t:.1f}' for t in tick_times[1:]])


        ## Cross-condition significance analysis per time
        use_paired = False                      # or True if trials are 1–1 matched
        sig_masks  = defaultdict(dict)          # key = (target_idx, i_cs) → bool[T]

        n_time = next(cs.shape[1] for cs in data_by_trial_filter_all[trial_filters[0]]['continuous_states_by_target'] if len(cs))

        for i_tgt in range(n_targets):
            cs_fast = data_by_trial_filter_all[trial_filters[0]]['continuous_states_by_target'][i_tgt]
            cs_slow = data_by_trial_filter_all[trial_filters[1]]['continuous_states_by_target'][i_tgt]
            if cs_fast.size == 0 or cs_slow.size == 0:        # no trials of this target
                continue

            # Shape:  (n_fast_trials,  T,  n_continuous_states)
            for i_cs in range(n_continuous_states):
                # ----- p-values per time bin ---------------------------------
                pvals = np.empty(n_time)
                for t in range(n_time):
                    pvals[t] = _p_value(cs_fast[:, t, i_cs],
                                        cs_slow[:, t, i_cs],
                                        use_paired)

                # ----- multiple-comparison correction (optional) -------------
                if correction_method is not None:
                    reject, _, _, _ = multipletests(
                        pvals, alpha=significance_alpha, method=correction_method)
                    sig_mask = reject
                else:
                    sig_mask = pvals < significance_alpha

                sig_masks[(i_tgt, i_cs)] = sig_mask

        ## Overlay horizontal bars where mask == True
        for i_tgt in range(n_targets):
            for i_cs in range(n_continuous_states):
                mask   = sig_masks.get((i_tgt, i_cs), np.zeros(n_time, dtype=bool))
                blocks = _find_contiguous_true(mask)
                if not blocks:
                    continue

                ax  = axs[i_cs, i_tgt]
                y0, y1 = ax.get_ylim()
                base   = y1 + 0.03 * (y1 - y0)    # 3 % above current top

                for s, e in blocks:
                    ax.hlines(base, s, e, color='k', linewidth=size_line_thin, alpha=alpha_line)

                # make sure the bar remains inside the axes
                ax.set_ylim(y0, base + 0.02 * (y1 - y0))
        
        ## Save fig
        fig.tight_layout()

        session_vis_dir = os.path.join(vis_dir, session_data_name)
        if not os.path.isdir(session_vis_dir):
            os.makedirs(session_vis_dir)
        save_path = os.path.join(session_vis_dir, res_name + '.pdf')
        fig.savefig(save_path, dpi=600, transparent=True, bbox_inches=None)

        ## Record significance counts
        for i_tgt in range(n_targets):
            for i_cs in range(n_continuous_states):

                if (i_tgt, i_cs) not in sig_masks:
                    raise ValueError(f"Significance mask not found for target {i_tgt} and continuous state {i_cs}")

                if sig_masks[(i_tgt, i_cs)][:peak_index].any():
                    sig_counts_before_peak[i_session, i_cs, i_tgt] += 1
                if sig_masks[(i_tgt, i_cs)][peak_index:].any():
                    sig_counts_after_peak[i_session, i_cs, i_tgt] += 1

    if save_significance_data:
        np.savez(
            os.path.join(results_dir, res_name + "_sig_counts.npz"),
            sig_counts_before_peak=sig_counts_before_peak,
            sig_counts_after_peak=sig_counts_after_peak
        )

    # ------------------------------------------------------------------
    # Bar plot of significance counts per latent dimension
    # ------------------------------------------------------------------
    # 1)  collapse session and target axes  →  one count per dimension
    sig_counts_before_peak_dim = sig_counts_before_peak.sum(axis=(0, 2))   # shape = (n_continuous_states,)
    sig_counts_after_peak_dim  = sig_counts_after_peak.sum(axis=(0, 2))

    # 2)  bar positions and width
    x        = np.arange(n_continuous_states)          # one group per dimension
    bar_w    = 0.38
    fig_sig, ax_sig = plt.subplots(figsize=(48 * mm, 30 * mm))

    ax_sig.bar(x - bar_w / 2,
               sig_counts_before_peak_dim,
               width=bar_w,
               color="tab:blue",
               label="Before peak")

    ax_sig.bar(x + bar_w / 2,
               sig_counts_after_peak_dim,
               width=bar_w,
               color="tab:orange",
               label="After peak")

    # 3)  cosmetics
    ax_sig.set_xticks(x)
    ax_sig.set_xticklabels([f"{i}" for i in range(n_continuous_states)])
    ax_sig.set_xlabel("Latent dimension index")
    ax_sig.set_ylabel("# significant samples")
    ax_sig.set_title("Significance prevalence (before vs. after peak)")
    ax_sig.set_ylim(0, max(sig_counts_before_peak_dim.max(),
                           sig_counts_after_peak_dim.max()) * 1.15)
    ax_sig.legend(frameon=False)
    fig_sig.tight_layout()

    # 4)  save and close
    save_path = os.path.join(vis_dir, res_name + "_sig_counts.pdf")
    fig_sig.savefig(save_path, dpi=600, transparent=True, bbox_inches=None)
    plt.close(fig_sig)


def plot_neural_speed_single_session(
    session_data_name: str,
    unit_filter: str,
    window_config: str,
    time_step: float,
    trial_filters: list[str],
    truncate_percentile: int = 90,
    pre_start_time_buffer: float = 0.2,
    post_reach_time_buffer: float = 0.5,
    superdiagonal_order: int = 1,
    time_marker: float | None = None,
    save_stats: bool = False,
    per_target_stats: bool = False,
    small_format: bool = False,
    peak_onset_time_nstd: float = 1.5,
):
    """Plot cross‑validated neural speed for two time bases in one figure.

    The left panel shows data that keep the *start of the trial* (``truncate_end``),
    the right panel shows data aligned to *reach* (``truncate_front``).  Both
    panels share the same y‑axis (scientific notation, ×10ⁿ), but the right‑hand
    y‑axis is hidden so the two panels have identical plotting areas.
    """
    if small_format:
        fig, axs = plt.subplots(1, 2, figsize=(45 * mm, 20 * mm), sharey=True, sharex=False)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(90 * mm, 45 * mm), sharey=True, sharex=False)

    pre_start_idx = int(pre_start_time_buffer / time_step)
    post_reach_idx = int(post_reach_time_buffer / time_step)

    stats = {}

    task_name = session_data_name.split("_")[-1]
    color_palettes_ = color_palettes[task_name]

    # Panel‑specific settings --------------------------------------------------
    formats = ["truncate_end", "truncate_front"]

    for i_ax, data_format in enumerate(formats):
        ax = axs[i_ax]

        for i_tf, trial_filter in enumerate(trial_filters):
            color = color_palettes_[trial_filter][1]

            # Load / compute crossnobis ----------------------------------------
            xnb_stack, _ = _load_or_compute_within_crossnobis(
                session_data_name, unit_filter, window_config,
                trial_filter, data_format, time_step,
                truncate_percentile, pre_start_idx, post_reach_idx,
                per_target_stack=per_target_stats)

            xnb_mat = np.mean(xnb_stack, axis=0)
            reach_marker_idx = xnb_mat.shape[0] - post_reach_idx

            diag = np.diag(xnb_mat, k=superdiagonal_order) / superdiagonal_order
            diag_stack = np.array(
                [np.diag(m, k=superdiagonal_order) / superdiagonal_order for m in xnb_stack]
            )
            sem = np.std(diag_stack, axis=0, ddof=1) / np.sqrt(diag_stack.shape[0])
            ci95 = t.ppf(0.975, df=xnb_stack.shape[0] - 1) * sem

            # Time axis -------------------------------------------------------
            t_axis = np.arange(diag.shape[0], dtype=float)
            if data_format == "truncate_front":
                t_axis = t_axis - diag.shape[0] + post_reach_idx
                reach_marker_plot = 0
            else:
                t_axis = t_axis - pre_start_idx
                reach_marker_plot = reach_marker_idx
            t_axis *= time_step

            # Save peak stats (truncate_end only) ----------------------------
            if save_stats and data_format == "truncate_end":
                stats[trial_filter] = utils_vis.compute_neural_speed_peak_onset_and_duration(
                    diag_stack,
                    min_amplitude=5,
                    time_step=time_step,
                    pre_start_time_buffer=pre_start_time_buffer,
                    peak_onset_time_nstd=peak_onset_time_nstd,
                )

            # Plot mean ±95 % CI ---------------------------------------------
            if not small_format:
                ax.plot(t_axis, diag, color=color, lw=size_line, alpha=alpha_line)
            else:
                ax.plot(t_axis, diag, color=color, lw=size_line_thin, alpha=alpha_line)
            ax.fill_between(
                t_axis,
                diag - sem,
                diag + sem,
                color=color,
                alpha=alpha_line_thin,
            )

            # Reach marker (red dashed) --------------------------------------
            if data_format == "truncate_front" and i_tf == 0:
                ax.axvline(
                    x=reach_marker_plot * time_step,
                    color="red",
                    ls=":",
                    lw=size_line_thin,
                    alpha=alpha_line,
                )

        # X‑axis --------------------------------------------------------------
        if not small_format:
            ax.set_xlabel("Time (s)", fontsize=5)
        ax.set_ylim([-500, 8000])

        # Pre‑start marker only on left panel (makes no sense on truncate_front)
        if data_format == "truncate_end" and pre_start_time_buffer > 0:
            ax.axvline(x=0, color="red", ls="--", lw=size_line_thin, alpha=alpha_line)

        # Optional visual delay marker on left panel
        if data_format == "truncate_end" and time_marker is not None:
            ax.axvline(x=time_marker, color="black", ls="--", lw=size_line_thin, alpha=alpha_line)

    # --------------------------------------------------------------------
    # Shared y‑axis formatting (scientific notation).  Right panel’s visible
    # y‑axis is hidden, but sharey=True guarantees limits & formatter match.
    # --------------------------------------------------------------------
    sf = ticker.ScalarFormatter(useMathText=True)
    sf.set_powerlimits((3, 3))
    axs[0].yaxis.set_major_formatter(sf)
    axs[0].ticklabel_format(axis="y", style="sci", scilimits=(3, 3))

    if not small_format:
        axs[0].set_ylabel(r"Cross-validated neural speed (($\Delta$Hz / s)$^2$)", fontsize=5)
    axs[0].spines["right"].set_visible(False)
    axs[0].spines["top"].set_visible(False)

    # Hide right‑hand y‑axis completely (ticks, labels, spine, offset text)
    axs[1].get_yaxis().set_visible(False)
    axs[1].yaxis.offsetText.set_visible(False)
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["left"].set_visible(False)
    axs[1].spines["right"].set_visible(False)

    # --------------------------------------------------------------------
    # Legend (use right panel so it doesn’t collide with y‑label)
    # --------------------------------------------------------------------
    # custom_lines = [
    #     Line2D([0], [0], color=color_palettes_[trial_filters[0]][1], lw=size_line, alpha=alpha_line),
    #     Line2D([0], [0], color=color_palettes_[trial_filters[1]][1], lw=size_line, alpha=alpha_line),
    # ]
    # axs[1].legend(custom_lines, ["ballistic", "sustained"], fontsize=5, fancybox=False, shadow=False)

    # --------------------------------------------------------------------
    # Save figure / stats
    # --------------------------------------------------------------------
    session_vis_dir = os.path.join(vis_dir, session_data_name)
    os.makedirs(session_vis_dir, exist_ok=True)

    save_parts = [
        session_data_name,
        "neural_speed_split",  # indicate two‑panel version
        unit_filter,
        window_config,
        f"superdiagonal{superdiagonal_order}",
        'nstd' + str(peak_onset_time_nstd)
    ]

    if save_stats:
        stats_name = "_".join(save_parts + ["stats"])
        with open(os.path.join(session_vis_dir, f"{stats_name}.pkl"), "wb") as f:
            pickle.dump(stats, f)

    if time_marker is not None:
        save_parts.append(f"tm{time_marker}")

    img_name = "_".join(save_parts)
    fig.tight_layout()
    fig.savefig(os.path.join(session_vis_dir, f"{img_name}.pdf"), dpi=600, transparent=True)

    plt.close(fig)


def plot_neural_speed(
    session_data_names: str,
    unit_filter: str,
    window_config: str,
    time_step: float,
    trial_filters: list[str],
    truncate_percentile: int = 90,
    pre_start_time_buffer: float = 0.2,
    post_reach_time_buffer: float = 0.5,
    superdiagonal_order: int = 1,
    time_marker: float | None = None,
    visual_delay_time: float = 0.15,
    peak_time: float = 0.5,
    supplement_format: bool = False,
    per_target: bool = False,
):
    """Plot cross‑validated neural speed for two time bases in one figure.

    The left panel shows data that keep the *start of the trial* (``truncate_end``),
    the right panel shows data aligned to *reach* (``truncate_front``).  Both
    panels share the same y‑axis (scientific notation, ×10ⁿ), but the right‑hand
    y‑axis is hidden so the two panels have identical plotting areas.
    """
    if supplement_format:
        fig, axs = plt.subplots(1, 2, figsize=(45*mm, 22.5*mm), sharey=True, sharex=False)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(90*mm, 45*mm), sharey=True, sharex=False)

    all_diag_vals = []
    pre_start_idx = int(pre_start_time_buffer / time_step)
    post_reach_idx = int(post_reach_time_buffer / time_step)

    task_name = session_data_names[0].split("_")[-1]
    color_palettes_ = color_palettes[task_name]

    # Panel‑specific settings --------------------------------------------------
    formats = ["truncate_end", "truncate_front"]

    for i_ax, data_format in enumerate(formats):
        ax = axs[i_ax]

        crossnobis_matrix_all = [[] for _ in trial_filters]
        min_lengths           = [np.inf] * len(trial_filters)

        # ─────────────── gather matrices over sessions ───────────────
        for session_data_name in session_data_names:
            for i_tf, trial_filter in enumerate(trial_filters):
                xnb_stack, _ = _load_or_compute_within_crossnobis(
                    session_data_name, unit_filter, window_config,
                    trial_filter, data_format, time_step,
                    truncate_percentile, pre_start_idx, post_reach_idx,
                    per_target=per_target)

                crossnobis_matrix_all[i_tf].append(xnb_stack)
                min_lengths[i_tf] = min(min_lengths[i_tf], xnb_stack.shape[-1])

        # ─────────────── truncate to common size & average ───────────────
        for i_tf in range(len(trial_filters)):
            for i in range(len(crossnobis_matrix_all[i_tf])):
                session_crossnobis_matrices = crossnobis_matrix_all[i_tf][i]
                if data_format == "truncate_end":
                    crossnobis_matrix_all[i_tf][i] = session_crossnobis_matrices[:, :min_lengths[i_tf], :min_lengths[i_tf]]
                elif data_format == "truncate_front":
                    crossnobis_matrix_all[i_tf][i] = session_crossnobis_matrices[:, -min_lengths[i_tf]:, -min_lengths[i_tf]:]

        crossnobis_matrix_all = [np.concatenate(mats, axis=0) for mats in crossnobis_matrix_all]

        # ipdb.set_trace()  # Debugging point to check the matrices

        # all_vals = np.concatenate([m.flatten() for m in crossnobis_matrix_all])
        # crossnobis_vmax = np.max(all_vals)

        for i_tf, trial_filter in enumerate(trial_filters):
            color = color_palettes_[trial_filter][1]

            reach_marker_idx = int(min_lengths[i_tf]) - post_reach_idx

            crossnobis_mean = np.mean(crossnobis_matrix_all[i_tf], axis=0)
            diag = np.diag(crossnobis_mean, k=superdiagonal_order) / superdiagonal_order
            all_diag_vals.append(diag)

            diag_stack = np.array(
                [np.diag(m, k=superdiagonal_order) / superdiagonal_order for m in crossnobis_matrix_all[i_tf]]
            )
            sem = np.std(diag_stack, axis=0, ddof=1) / np.sqrt(diag_stack.shape[0])
            ci95 = t.ppf(0.975, df=diag_stack.shape[0] - 1) * sem

            # Time axis -------------------------------------------------------
            t_axis = np.arange(diag.shape[0], dtype=float)
            if data_format == "truncate_front":
                t_axis = t_axis - diag.shape[0] + post_reach_idx
                reach_marker_plot = 0
            else:
                t_axis = t_axis - pre_start_idx
                reach_marker_plot = reach_marker_idx
            t_axis *= time_step

            # Plot mean ±95 % CI ---------------------------------------------
            ax.plot(t_axis, diag, color=color, lw=size_line_thin, alpha=alpha_line)
            ax.fill_between(
                t_axis,
                diag - sem,
                diag + sem,
                color=color,
                alpha=alpha_line_thin,
            )

        # X‑axis --------------------------------------------------------------
        if not supplement_format:
            ax.set_xlabel("Time (s)", fontsize=5)
        ax.set_ylim([-1000, 9000])
        # ax.set_ylim([-1000, 11000])

        # Pre‑start marker only on left panel (makes no sense on truncate_front)
        if data_format == "truncate_end" and pre_start_time_buffer > 0:
            ax.axvline(x=0, color="red", ls="--", lw=size_line_thin, alpha=alpha_line)

        # Optional visual delay marker on left panel
        if data_format == "truncate_end" and time_marker is not None:
            ax.axvline(x=time_marker, color="black", ls="--", lw=size_line_thin, alpha=alpha_line)

        # Reach marker (red dashed) --------------------------------------
        if data_format == "truncate_front":
            ax.axvline(x=reach_marker_plot * time_step, color="red", ls=":", lw=size_line_thin, alpha=alpha_line)

        ax.tick_params(labelsize=5)

    # --------------------------------------------------------------------
    # Shared y‑axis formatting (scientific notation).  Right panel’s visible
    # y‑axis is hidden, but sharey=True guarantees limits & formatter match.
    # --------------------------------------------------------------------
    sf = ticker.ScalarFormatter(useMathText=True)
    sf.set_powerlimits((3, 3))
    axs[0].yaxis.set_major_formatter(sf)
    axs[0].ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
    axs[0].yaxis.offsetText.set_fontsize(5)

    if not supplement_format:
        axs[0].set_ylabel(r"Cross-validated neural speed (($\Delta$Hz / s)$^2$)", fontsize=5)
    axs[0].spines["right"].set_visible(False)
    axs[0].spines["top"].set_visible(False)

    # Hide right‑hand y‑axis completely (ticks, labels, spine, offset text)
    axs[1].get_yaxis().set_visible(False)
    axs[1].yaxis.offsetText.set_visible(False)
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["left"].set_visible(False)
    axs[1].spines["right"].set_visible(False)

    # --------------------------------------------------------------------
    # Legend (use right panel so it doesn’t collide with y‑label)
    # --------------------------------------------------------------------
    # custom_lines = [
    #     Line2D([0], [0], color=color_palettes_[trial_filters[0]][1], lw=size_line, alpha=alpha_line),
    #     Line2D([0], [0], color=color_palettes_[trial_filters[1]][1], lw=size_line, alpha=alpha_line),
    # ]
    # axs[1].legend(custom_lines, ["ballistic", "sustained"], fontsize=5, fancybox=False, shadow=False)

    # --------------------------------------------------------------------
    # Save figure / stats
    # --------------------------------------------------------------------
    img_parts = [
        task_name,
        "neural_speed_split",  # indicate two‑panel version
        unit_filter,
        window_config,
        f"superdiag{superdiagonal_order}",
    ]
    if time_marker is not None:
        img_parts.append(f"tm{time_marker}")

    img_name = "_".join(img_parts)
    fig.tight_layout()
    all_diag_concat = np.concatenate(all_diag_vals)
    print(f"[plot_neural_speed] {img_name}: data [{float(np.min(all_diag_concat)):.1f}, {float(np.max(all_diag_concat)):.1f}]")
    fig.savefig(os.path.join(vis_dir, f"{img_name}.pdf"), dpi=600, transparent=True, bbox_inches=None)
    plt.close(fig)

    
def plot_time_crossnobis_RDM_matrix(
    session_data_names: list[str],
    unit_filter: str,
    window_config: str,
    time_step: float,
    data_format: str,
    trial_filters: list[str],
    truncate_percentile: int = 90,
    pre_start_time_buffer: float = 0.2,
    post_reach_time_buffer: float = 0.5,
    visual_delay_time: float = 0.15,
    peak_time: float = 0.5,
    supplement_format: bool = False,
    per_target: bool = False,
    show_phase_bars: bool = True,
):
    """
    Plot average time-time Cross-Nobis RDMs for the given sessions and trial
    filters, save the heat-map pair as <…>.pdf and a separate horizontal
    colour-bar as <…>_COLORBAR.pdf in the same folder.
    """

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_greys", ["white", "#606060"])
    task_name = session_data_names[0].split("_")[-1]

    # ─────────────── figure canvas for the two heat-maps ───────────────
    if supplement_format:
        fig, axs = plt.subplots(1, 2, figsize=(45 * mm, 22.5 * mm))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(90 * mm, 45 * mm))

    # collecting per-filter matrices
    crossnobis_matrix_all = [[] for _ in trial_filters]
    min_lengths           = [np.inf] * len(trial_filters)

    pre_start_idx  = int(pre_start_time_buffer  / time_step)
    post_reach_idx = int(post_reach_time_buffer / time_step)

    # ─────────────── gather matrices over sessions ───────────────
    for session_data_name in session_data_names:
        for i_tf, trial_filter in enumerate(trial_filters):
            mats, _ = _load_or_compute_within_crossnobis(
                session_data_name, unit_filter, window_config,
                trial_filter, data_format, None, truncate_percentile,
                pre_start_idx, post_reach_idx,
                per_target=per_target, trim_buffers=False)

            crossnobis_matrix_all[i_tf].append(mats)
            min_lengths[i_tf] = min(min_lengths[i_tf], mats.shape[1])

    # ─────────────── truncate to common size & average ───────────────
    for i_tf in range(len(trial_filters)):
        for i in range(len(crossnobis_matrix_all[i_tf])):
            m = crossnobis_matrix_all[i_tf][i]
            crossnobis_matrix_all[i_tf][i] = m[:, :min_lengths[i_tf], :min_lengths[i_tf]]

    crossnobis_matrix_all = [
        np.mean(np.concatenate(mats, axis=0), axis=0) for mats in crossnobis_matrix_all
    ]

    # ─── downsample by 10 for 10 ms resolution ───
    ds = 10
    crossnobis_matrix_all = [m[::ds, ::ds] for m in crossnobis_matrix_all]
    min_lengths = [int(l) // ds for l in min_lengths]
    plot_time_step = time_step * ds
    plot_pre_start_idx = pre_start_idx // ds
    plot_post_reach_idx = post_reach_idx // ds

    all_vals = np.concatenate([m.flatten() for m in crossnobis_matrix_all])
    data_vmin = float(np.min(all_vals))
    data_vmax = float(np.max(all_vals))

    crossnobis_vmax = 200 if supplement_format else 100

    # ─────────────── draw heat-maps (no colour-bar) ───────────────
    for i_tf, (trial_filter, ax) in enumerate(zip(trial_filters, axs)):
        mat  = crossnobis_matrix_all[i_tf]
        N    = mat.shape[0]

        # build time axis (in bin indices) then shift to seconds
        t_bins = np.arange(N, dtype=float)
        if data_format == "truncate_front":
            t_bins -= N - plot_post_reach_idx
            reach_marker = N - plot_post_reach_idx
        else:
            t_bins -= plot_pre_start_idx
            reach_marker = N - 1  # just so variable exists

        # heat-map
        im = sns.heatmap(
            mat,
            ax=ax,
            cmap=cmap,
            vmin=0, vmax=crossnobis_vmax,
            square=True,
            cbar=False,                     # ❶ NO colour-bar here
            linewidths=0,
            linecolor=None,
            rasterized=True,
        )
        # remove faint grid that sometimes appears in vector backends
        im.collections[0].set_edgecolor("face")
        im.collections[0].set_linewidth(0)

        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()

        # nice tick labels every ~10 bins
        tick_t_intervals = np.array([0.2, 0.5])
        if supplement_format:
            tick_t_intervals *= 2
        tick_idx_intervals = (tick_t_intervals / plot_time_step).astype(int)
        tick_idx  = np.arange(plot_pre_start_idx, N, tick_idx_intervals[i_tf])
        tick_lbls = [f"{t_bins[i] * plot_time_step:.1f}" for i in tick_idx]
        ax.set_xticks(tick_idx); ax.set_xticklabels(tick_lbls, rotation=0)
        ax.set_yticks(tick_idx); ax.set_yticklabels(tick_lbls)
        if not supplement_format:
            ax.set_xlabel("Time (s)", fontsize=7)
            ax.set_ylabel("Time (s)", fontsize=7)

        # helper to draw corner-truncated markers (vertical & horizontal)
        def _draw_marker(idx, color, linestyle, lw):
            ax.plot([0, idx],   [idx, idx], color=color, lw=lw,
                    linestyle=linestyle, alpha=alpha_line)
            ax.plot([idx, idx], [0, idx],   color=color, lw=lw,
                    linestyle=linestyle, alpha=alpha_line)

        # start-of-trial marker
        marker_lw = size_line_thin
        if pre_start_time_buffer > 0 and data_format != "truncate_front":
            _draw_marker(plot_pre_start_idx, "red", "--", marker_lw)

        # reach marker
        if data_format != "truncate_end":
            _draw_marker(reach_marker, "red", "--", marker_lw)

        # visual-delay marker
        if data_format != "truncate_front":
            vis_idx = int(visual_delay_time / plot_time_step + plot_pre_start_idx)
            _draw_marker(vis_idx, "black", "--", marker_lw)

            vis_idx = int(peak_time / plot_time_step + plot_pre_start_idx)
            _draw_marker(vis_idx, "black", ":", marker_lw)

            vis_idx = min_lengths[i_tf] - plot_post_reach_idx
            _draw_marker(vis_idx, "red", ":", marker_lw)

        # ─── Phase bars along axes ──────────────────────────────────
        if not show_phase_bars:
            continue
        vis_idx_bar = int(visual_delay_time / plot_time_step + plot_pre_start_idx)
        peak_idx_bar = int(peak_time / plot_time_step + plot_pre_start_idx)
        reach_idx_bar = int(min_lengths[i_tf]) - plot_post_reach_idx
        palette = color_palettes[task_name][trial_filter]  # [light, mid, dark]

        phase_segs = [
            (plot_pre_start_idx,  vis_idx_bar,   palette[2]),
            (vis_idx_bar,    peak_idx_bar,  palette[1]),
            (peak_idx_bar,   reach_idx_bar, palette[0]),
        ]

        bar_thickness = 0.02
        divider = make_axes_locatable(ax)

        ax_xbar = divider.append_axes("bottom", size=bar_thickness, pad=0.02)
        ax_xbar.set_xlim(ax.get_xlim())
        ax_xbar.set_ylim(0, 1)
        for s0, s1, sc in phase_segs:
            ax_xbar.add_patch(Rectangle((s0, 0), s1 - s0, 1,
                                        facecolor=sc, edgecolor='none'))
        ax_xbar.set_xticks([]); ax_xbar.set_yticks([])
        for spine in ax_xbar.spines.values():
            spine.set_visible(False)

        ax_ybar = divider.append_axes("left", size=bar_thickness, pad=0.02)
        ax_ybar.set_ylim(ax.get_ylim())
        ax_ybar.set_xlim(0, 1)
        for s0, s1, sc in phase_segs:
            ax_ybar.add_patch(Rectangle((0, s0), 1, s1 - s0,
                                        facecolor=sc, edgecolor='none'))
        ax_ybar.set_xticks([]); ax_ybar.set_yticks([])
        for spine in ax_ybar.spines.values():
            spine.set_visible(False)

    # ─────────────── save main figure ───────────────
    img_name = "_".join(
        str(x)
        for x in (
            task_name,
            "crossnobis_RDM_matrix",
            unit_filter,
            window_config,
            data_format,
            "all_sessions",
        )
        if x
    )

    plt.rcParams["path.simplify"] = False
    main_path = os.path.join(vis_dir, f"{img_name}.pdf")

    fig.tight_layout()
    print(f"[plot_time_crossnobis_RDM_matrix] {img_name}: data [{data_vmin:.1f}, {data_vmax:.1f}]")
    fig.savefig(main_path, transparent=True, bbox_inches=None, dpi=600)
    plt.close(fig)


    # ───────────────  STAND-ALONE COLOUR-BAR  ───────────────
    # “thin” bar: long and short – 50 mm × 6 mm
    if supplement_format:
        fig_cbar, ax_cbar = plt.subplots(figsize=(22 * mm, 10 * mm))
    else:
        fig_cbar, ax_cbar = plt.subplots(figsize=(30 * mm, 12 * mm))

    sf = ticker.ScalarFormatter(useMathText=True)
    sf.set_powerlimits((3, 3))

    mappable = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=0, vmax=crossnobis_vmax)
    )
    mappable.set_array([])

    # create the bar
    cbar = fig_cbar.colorbar(
        mappable,
        cax=ax_cbar,
        orientation="horizontal",
    )
    # ticks & labels
    # cbar.ax.xaxis.set_major_formatter(sf)
    if not supplement_format:
        cbar.ax.tick_params(labelsize=5, length=0, width=0)
    else:
        cbar.set_ticks([])

    # put label above the bar
    cbar.set_label("Crossnobis distance (ΔHz$^2$)",
                   fontsize=5, labelpad=2, rotation=0)
    cbar.ax.xaxis.set_label_position("top")     # label above
    cbar.ax.xaxis.set_ticks_position("bottom")  # ticks remain below


    # ── remove the black outline & spines ───────────────────
    cbar.outline.set_visible(False)             # no frame around colour-bar
    for spine in ax_cbar.spines.values():       # no axis box either
        spine.set_visible(False)

    # tidy & save
    fig_cbar.subplots_adjust(left=0.03, right=0.97, top=0.75, bottom=0.60)
    bar_path = os.path.join(vis_dir, f"{img_name}_COLORBAR.pdf")
    fig_cbar.savefig(bar_path, transparent=True, dpi=600)
    plt.close(fig_cbar)


def plot_cross_condition_crossnobis_RDM_matrix(
    session_data_names,
    unit_filter,
    window_config,
    time_step,
    data_format,
    trial_filters,
    truncate_percentile=90,
    pre_start_time_buffer=0.2,
    post_reach_time_buffer=0.5,
    visual_delay_time=0.15,
    peak_time=0.5,
    per_target=False,
    show_phase_bars=True,
    supplement_format=False,
):
    """Plot a rectangular cross-condition crossnobis distance matrix.

    Rows correspond to time points from trial_filters[0] (e.g. fast/near)
    and columns to time points from trial_filters[1] (e.g. slow/far).
    If per_target is True, computes per-target matrices and averages them.
    """

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_greys", ["white", "#606060"])
    task_name = session_data_names[0].split('_')[-1]

    pre_start_idx  = int(pre_start_time_buffer  / time_step)
    post_reach_idx = int(post_reach_time_buffer / time_step)

    cross_matrices_all = []
    min_T1 = np.inf
    min_T2 = np.inf

    for session_data_name in session_data_names:
        cross_avg = _load_or_compute_cross_condition_crossnobis(
            session_data_name, unit_filter, window_config,
            trial_filters, data_format, time_step,
            truncate_percentile, pre_start_idx, post_reach_idx,
            per_target=per_target)

        if cross_avg is None:
            continue

        cross_matrices_all.append(cross_avg)
        min_T1 = min(min_T1, cross_avg.shape[0])
        min_T2 = min(min_T2, cross_avg.shape[1])

    if not cross_matrices_all:
        print(f'  Warning: no valid cross-condition matrices for '
              f'{session_data_names}, {unit_filter}, {trial_filters}. Skipping.')
        plt.close()
        return

    # Truncate to common size and average across sessions
    for i in range(len(cross_matrices_all)):
        cross_matrices_all[i] = cross_matrices_all[i][:min_T1, :min_T2]

    cross_matrix_avg = np.mean(cross_matrices_all, axis=0)
    data_vmin = float(np.min(cross_matrix_avg))
    data_vmax = float(np.max(cross_matrix_avg))

    # ─── downsample by 10 for 10 ms resolution ───
    ds = 10
    cross_matrix_avg = cross_matrix_avg[::ds, ::ds]
    min_T1 = int(min_T1) // ds
    min_T2 = int(min_T2) // ds
    plot_time_step = time_step * ds
    plot_pre_start_idx = pre_start_idx // ds
    plot_post_reach_idx = post_reach_idx // ds

    # Plot the rectangular heatmap
    if supplement_format:
        fig, ax = plt.subplots(1, 1, figsize=(60*mm, 45*mm))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(90*mm, 45*mm))

    crossnobis_vmax = 200 if supplement_format else 100

    im = sns.heatmap(
        cross_matrix_avg,
        ax=ax,
        cmap=cmap,
        vmin=0, vmax=crossnobis_vmax,
        cbar=False,
        linewidths=0,
        linecolor=None,
        rasterized=True,
    )
    im.collections[0].set_edgecolor("face")
    im.collections[0].set_linewidth(0)

    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()

    # Tick labels — 0.2s spacing on both axes
    T1, T2 = int(min_T1), int(min_T2)
    tick_interval = int(0.2 / plot_time_step)

    tick_idx_y = np.arange(plot_pre_start_idx, T1, tick_interval)
    tick_lbls_y = [f"{(i - plot_pre_start_idx) * plot_time_step:.1f}" for i in tick_idx_y]
    ax.set_yticks(tick_idx_y)
    ax.set_yticklabels(tick_lbls_y)

    tick_idx_x = np.arange(plot_pre_start_idx, T2, tick_interval)
    tick_lbls_x = [f"{(i - plot_pre_start_idx) * plot_time_step:.1f}" for i in tick_idx_x]
    ax.set_xticks(tick_idx_x)
    ax.set_xticklabels(tick_lbls_x, rotation=0)

    if task_name == 'RadialGrid':
        label_map_rg = {'near': 'Near', 'far': 'Far'}
        ax.set_ylabel(f'{label_map_rg.get(trial_filters[0], trial_filters[0].capitalize())} trial time (s)', fontsize=7)
        ax.set_xlabel(f'{label_map_rg.get(trial_filters[1], trial_filters[1].capitalize())} trial time (s)', fontsize=7)
    else:
        label_map = {'fast': 'Ballistic', 'slow': 'Sustained'}
        ax.set_ylabel(f'{label_map.get(trial_filters[0], trial_filters[0])} trial time (s)', fontsize=7)
        ax.set_xlabel(f'{label_map.get(trial_filters[1], trial_filters[1])} trial time (s)', fontsize=7)

    # helper to draw corner-truncated markers (lines meet at their intersection)
    marker_lw = size_line_thin

    def _draw_marker(row_idx, col_idx, color, linestyle, lw):
        ax.plot([0, col_idx], [row_idx, row_idx], color=color, lw=lw,
                linestyle=linestyle, alpha=alpha_line)
        ax.plot([col_idx, col_idx], [0, row_idx], color=color, lw=lw,
                linestyle=linestyle, alpha=alpha_line)

    # Trial start marker (red dashed)
    if pre_start_time_buffer > 0:
        _draw_marker(plot_pre_start_idx, plot_pre_start_idx, 'red', '--', marker_lw)

    # Reach end markers (red dotted) — different for each condition
    reach_idx_rows = T1 - plot_post_reach_idx
    reach_idx_cols = T2 - plot_post_reach_idx
    _draw_marker(reach_idx_rows, reach_idx_cols, 'red', ':', marker_lw)

    # Visual delay markers (black dashed)
    if pre_start_time_buffer > 0:
        vis_idx = int(visual_delay_time / plot_time_step + plot_pre_start_idx)
        _draw_marker(vis_idx, vis_idx, 'black', '--', marker_lw)

        # Peak time markers (black dotted)
        peak_idx = int(peak_time / plot_time_step + plot_pre_start_idx)
        _draw_marker(peak_idx, peak_idx, 'black', ':', marker_lw)

    # ─── Phase bars along axes ───────────────────────────────────────
    if show_phase_bars:
        vis_idx = int(visual_delay_time / plot_time_step + plot_pre_start_idx)
        peak_idx = int(peak_time / plot_time_step + plot_pre_start_idx)
        reach_idx_rows = T1 - plot_post_reach_idx
        reach_idx_cols = T2 - plot_post_reach_idx

        # Phase boundaries and colors for each axis
        # Y-axis → trial_filters[0], X-axis → trial_filters[1]
        palette_y = color_palettes[task_name][trial_filters[0]]  # [light, mid, dark]
        palette_x = color_palettes[task_name][trial_filters[1]]

        def _phase_segments(reach_idx, palette):
            """Return list of (start, end, color) for phase bar segments."""
            return [
                (plot_pre_start_idx, vis_idx,    palette[2]),      # visual delay
                (vis_idx,    peak_idx,            palette[1]),      # before peak
                (peak_idx,   reach_idx,           palette[0]),      # after peak
            ]

        bar_thickness = 0.02  # fraction of axes size

        # Bottom bar (x-axis phases for trial_filters[1])
        divider = make_axes_locatable(ax)
        ax_xbar = divider.append_axes("bottom", size=bar_thickness, pad=0.02)
        ax_xbar.set_xlim(ax.get_xlim())
        ax_xbar.set_ylim(0, 1)
        for seg_start, seg_end, seg_color in _phase_segments(reach_idx_cols, palette_x):
            ax_xbar.add_patch(Rectangle(
                (seg_start, 0), seg_end - seg_start, 1,
                facecolor=seg_color, edgecolor='none'))
        ax_xbar.set_xticks([])
        ax_xbar.set_yticks([])
        for spine in ax_xbar.spines.values():
            spine.set_visible(False)

        # Left bar (y-axis phases for trial_filters[0])
        ax_ybar = divider.append_axes("left", size=bar_thickness, pad=0.02)
        ax_ybar.set_ylim(ax.get_ylim())
        ax_ybar.set_xlim(0, 1)
        for seg_start, seg_end, seg_color in _phase_segments(reach_idx_rows, palette_y):
            ax_ybar.add_patch(Rectangle(
                (0, seg_start), 1, seg_end - seg_start,
                facecolor=seg_color, edgecolor='none'))
        ax_ybar.set_xticks([])
        ax_ybar.set_yticks([])
        for spine in ax_ybar.spines.values():
            spine.set_visible(False)

    # Save
    per_target_str = 'per_target' if per_target else 'pooled'
    img_name = '_'.join(
        str(x) for x in (
            task_name,
            'cross_condition_crossnobis_RDM',
            unit_filter,
            window_config,
            data_format,
            per_target_str,
            'all_sessions',
        ) if x
    )
    save_path = os.path.join(vis_dir, f"{img_name}.pdf")
    fig.tight_layout()
    print(f"[plot_cross_condition_crossnobis_RDM_matrix] {img_name}: data [{data_vmin:.1f}, {data_vmax:.1f}]")
    fig.savefig(save_path, transparent=True, bbox_inches=None, dpi=600)
    plt.close(fig)

    # Standalone colorbar
    if supplement_format:
        fig_cbar, ax_cbar = plt.subplots(figsize=(22*mm, 10*mm))
    else:
        fig_cbar, ax_cbar = plt.subplots(figsize=(30*mm, 12*mm))
    mappable = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=0, vmax=crossnobis_vmax))
    mappable.set_array([])
    cbar = fig_cbar.colorbar(mappable, cax=ax_cbar, orientation="horizontal")
    if not supplement_format:
        cbar.ax.tick_params(labelsize=5, length=0, width=0)
    else:
        cbar.set_ticks([])
    cbar.set_label("Crossnobis distance (ΔHz$^2$)",
                   fontsize=5, labelpad=2, rotation=0)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.outline.set_visible(False)
    for spine in ax_cbar.spines.values():
        spine.set_visible(False)
    fig_cbar.subplots_adjust(left=0.05, right=0.95, top=0.75, bottom=0.60)
    bar_path = os.path.join(vis_dir, f"{img_name}_COLORBAR.pdf")
    fig_cbar.savefig(bar_path, transparent=True, dpi=600)
    plt.close(fig_cbar)


def plot_time_crossnobis_RDM_matrix_single_session(
    session_data_name,
    unit_filter, 
    data_format,
    truncate_percentile=10,
    pre_start_time_buffer=0.2,
    post_reach_time_buffer=0.5,
    ci_min=False,
    ci_max=False,
    pval=False,
    signed_square_root=True,
    time_marker=None):

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    pre_start_index_buffer = int(pre_start_time_buffer / config.time_step)
    post_reach_index_buffer = int(post_reach_time_buffer / config.time_step)

    for i_tf, trial_filter in enumerate(trial_filters):

        # Load data
        data_loader = utils_processing.DataLoader(
            data_dir,
            results_dir,
            session_data_name,
            unit_filter,
            None,
            window_config,
            trial_filter)

        data_loader.load_firing_rate_data()

        firing_rates_simple, _, _, _, _, _ = data_loader.reformat_firing_rate_data(
            data_format, 
            index_buffer=post_reach_index_buffer,
            truncate_percentile=truncate_percentile)
        
        # Remove post reach buffer
        firing_rates_simple = firing_rates_simple[:, :-post_reach_index_buffer, :]

        # Compute crossnobis matrix
        crossnobis_matrix, crossnobis_matrices = utils_processing.compute_crossnobis_matrix(
            firing_rates_simple, 
            signed_square_root=signed_square_root)

        if ci_min or ci_max:
            mean_rdm = np.mean(crossnobis_matrices, axis=0)  # Mean across trials
            std_rdm = np.std(crossnobis_matrices, axis=0, ddof=1)  # Standard deviation
            se_rdm = std_rdm / np.sqrt(crossnobis_matrices.shape[0])  # Standard error

            if ci_min:
                mat = mean_rdm - 1.96 * se_rdm
                cmap = sns.color_palette("light:seagreen", as_cmap=True)
                cbar_label = 'Lower Bound of 95% CI of Crossnobis Distance ($\Delta$Hz$^2$)'
            else:
                mat = mean_rdm + 1.96 * se_rdm
                cmap = sns.color_palette("light:salmon", as_cmap=True)
                cbar_label = 'Upper Bound of 95% CI of Crossnobis Distance ($\Delta$Hz$^2$)'

            norm = None
            vmin = 0
            marker_color = 'black'
            cbar_label_size = label_fontsize - 4
            formatter = None

        elif pval:
            mean_rdm = np.mean(crossnobis_matrices, axis=0)
            trial_length = crossnobis_matrices.shape[1]
            mat = np.zeros((trial_length, trial_length))

            for i in range(trial_length):
                for j in range(trial_length):
                    if i == j:
                        mat[i, j] = 1
                        continue
                    # Extract the distribution for cell (i, j)
                    dist_vals = crossnobis_matrices[:, i, j]
                    # Perform one-sample t-test
                    _, mat[i, j] = ttest_1samp(dist_vals, 0)

            cmap = sns.color_palette("light:salmon", as_cmap=True)
            norm = mcolors.LogNorm(vmin=1e-10, vmax=mat.max())
            vmin = 1e-10
            marker_color = 'black'
            cbar_label = 'p-value (log scale)'
            cbar_label_size = label_fontsize - 2
            formatter = ticker.FuncFormatter(utils_vis.sci_notation_fmt)
        
        else:
            mat = crossnobis_matrix
            cmap = 'viridis'
            norm = None
            vmin = 0
            marker_color = 'white'
            cbar_label = 'Crossnobis Distance ($\Delta$Hz$^2$)'
            cbar_label_size = label_fontsize - 2
            formatter = None

        assert mat.shape == crossnobis_matrix.shape

        # Adjust x-axis 
        x_axis_data = np.arange(mat.shape[0]).astype(float)

        if data_format == 'truncate_front':
            x_axis_data = x_axis_data - mat.shape[0] + post_reach_index_buffer
            reach_time_marker = mat.shape[0] - post_reach_index_buffer
        else:
            x_axis_data = x_axis_data - pre_start_index_buffer


        sns.heatmap(
            mat,
            norm=norm,
            ax=axs[i_tf],
            cmap=cmap,
            vmin=vmin,
            # vmax=crossnobis_vmax,
            cbar_kws={'format': formatter},
            square=True,
        )
        
        cbar = axs[i_tf].collections[0].colorbar
        cbar.ax.set_ylabel(cbar_label, fontsize=cbar_label_size)
        
        axs[i_tf].invert_yaxis()
        
        tick_indices = np.arange(0, len(x_axis_data), 10)  # Indices of ticks you want to show
        tick_labels = [f"{(x_axis_data[i] * config.time_step):.1f}" for i in tick_indices]

        axs[i_tf].set_xticks(tick_indices)
        axs[i_tf].set_xticklabels(tick_labels, rotation=45, ha='right')
        axs[i_tf].set_yticks(tick_indices)
        axs[i_tf].set_yticklabels(tick_labels)
        axs[i_tf].set_xlabel('Time (s)', fontsize=label_fontsize)
        axs[i_tf].set_ylabel('Time (s)', fontsize=label_fontsize)

        ## Time markers
        # Start time marker
        if pre_start_time_buffer > 0 and data_format != 'truncate_front':
            axs[i_tf].axvline(
                x=pre_start_index_buffer, 
                color='red', 
                linestyle='--',
                linewidth=size_line,
                alpha=alpha_line)
            
            axs[i_tf].axhline(
                y=pre_start_index_buffer, 
                color='red', 
                linestyle='--',
                linewidth=size_line,
                alpha=alpha_line)

        # Reach time marker
        if data_format != 'truncate_end':
            axs[i_tf].axvline(
                x=reach_time_marker, 
                color='red', 
                linestyle='--',
                linewidth=size_line,
                alpha=alpha_line)
            
            axs[i_tf].axhline(
                y=reach_time_marker, 
                color='red', 
                linestyle='--',
                linewidth=size_line,
                alpha=alpha_line)
        
        # Visual delay time marker
        if data_format != 'truncate_front' and time_marker is not None:
            axs[i_tf].axvline(
                x=time_marker / config.time_step + pre_start_index_buffer, 
                color=marker_color, 
                linestyle='--',
                linewidth=size_line_thin,
                alpha=alpha_line)
            
            axs[i_tf].axhline(
                y=time_marker / config.time_step + pre_start_index_buffer, 
                color=marker_color, 
                linestyle='--',
                linewidth=size_line_thin,
                alpha=alpha_line)

        # axs.set_title(trial_filter)

    
    img_name = '_'.join(map(str, [x for x in [
        'crossnobis_RDM_matrix',
        unit_filter,
        window_config,
        data_format,
        trial_filter] if x is not None]))

    if signed_square_root:
        img_name += '_SSR'
    if time_marker is not None:
        img_name += '_tm' + str(time_marker)
    if ci_min:
        img_name += '_ci_min'
    if ci_max:
        img_name += '_ci_max'
    if pval:
        img_name += '_pval'

    session_vis_dir = os.path.join(vis_dir, session_data_name)
    if not os.path.isdir(session_vis_dir):
        os.makedirs(session_vis_dir)

    plt.rcParams['path.simplify'] = False

    save_path = os.path.join(session_vis_dir, img_name + '.pdf')

    fig.tight_layout()
    fig.savefig(save_path, dpi=600, transparent=True)
    plt.close()


def _find_contiguous_true(mask):
    """Utility: return list of (start_idx, end_idx) for contiguous True blocks."""
    blocks = []
    for k, g in itertools.groupby(enumerate(mask), key=lambda x: x[1]):
        if k:
            idxs = [i for i, _ in g]
            blocks.append((idxs[0], idxs[-1]))
    return blocks


def _p_value(vec1, vec2, use_paired):
    """Return two‑sided p between two 1‑D arrays (NaNs already removed)."""
    if use_paired:
        n_common = min(len(vec1), len(vec2))
        if n_common < 2:
            return 1.0
        return ttest_rel(vec1[:n_common], vec2[:n_common]).pvalue
    else:
        if len(vec1) < 1 or len(vec2) < 1:
            return 1.0
        # return ttest_ind(vec1, vec2, equal_var=False).pvalue
        return mannwhitneyu(vec1, vec2).pvalue


def plot_direction_tuned_neurons_single_session(
        session_data_name: str,
        unit_filter: str,
        data_format: str,
        target_tuned: np.ndarray | None,
        *,
        direction_tuned_neuron_indices: np.ndarray = None,
        trial_filters=("fast", "slow"),
        truncate_percentile=None,
        max_neurons_plot: int | None = None,
        visual_delay_time: float | None = 0.1,
        lock_row_ylim: bool = True,
        colour_palette=None,
        # NEW -----------------------------------------------------------------
        plot_significance: bool = False,
        significance_alpha: float = 0.05,
        correction_method: str = "fdr_bh",
):
    """Plot firing‑rate (or PC/dPC scores) per target direction.

    Parameters
    ----------
    ... (unchanged until *)
    plot_significance : bool, default False
        If True, perform per‑time‑bin statistical testing between conditions and
        mark segments where the difference is significant (after multiple‑
        comparison correction) with a translucent band.
    significance_alpha : float
        Family‑wise alpha to control after correction.
    correction_method : {"fdr_bh", "bonferroni", ...}
        Method string forwarded to :pyfunc:`statsmodels.stats.multitest.multipletests`.
    """

    # ------------------------------------------------------------------
    # --- Helper & sanity checks ---------------------------------------
    # ------------------------------------------------------------------
    assert data_format == "truncate_end"

    if colour_palette is None:
        # default 8‑target qualitative palette
        colour_palette = {
            0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728",
            4: "#9467bd", 5: "#8c564b", 6: "#e377c2", 7: "#7f7f7f",
        }

    visual_delay_index = int(visual_delay_time / config.time_step) if visual_delay_time else 0

    # ------------------------------------------------------------------
    # --- Load data ----------------------------------------------------
    # ------------------------------------------------------------------
    data_per_filter: dict[str, dict] = {}
    for trial_filter in trial_filters:
        data_loader = utils_processing.DataLoader(
            data_dir=data_dir,
            results_dir=results_dir,
            session_data_name=session_data_name,
            unit_filter=unit_filter,
            input_unit_filter=None,
            window_config=window_config,
            trial_filter=trial_filter,
        )
        data_loader.load_firing_rate_data()
        data_loader.load_cursor_data()
        data_loader.remove_target_overlap(target_radius=session_target_radii[session_data_name])

        target_ids = data_loader.get_target_ids()

        firing_rates_simple, _, _, _, _, trial_keep_mask = data_loader.reformat_firing_rate_data(
            data_format=data_format, trial_length_filter_percentile=truncate_percentile
        )
        target_ids = target_ids[trial_keep_mask]

        # visual delay shift if requested
        if visual_delay_index > 0:
            firing_rates_simple = firing_rates_simple[:, visual_delay_index:, :]

        # Keep only selected neurons
        if direction_tuned_neuron_indices is not None:
            firing_rates_simple = firing_rates_simple[:, :, direction_tuned_neuron_indices]

        data_per_filter[trial_filter] = {
            "target_ids": target_ids,
            "firing_rates": firing_rates_simple,
            "trial_length": firing_rates_simple.shape[1],
        }

    n_targets = len(np.unique(data_per_filter[trial_filters[0]]["target_ids"]))
    n_neurons = firing_rates_simple.shape[-1]
    fast_label, slow_label = trial_filters

    rows = n_neurons if max_neurons_plot is None else min(n_neurons, max_neurons_plot)
    if direction_tuned_neuron_indices is not None:
        row_labels = [f'Neuron {idx}' for idx in direction_tuned_neuron_indices]
    else:
        row_labels = [f'Neuron {i + 1}' for i in range(rows)]

    # ------------------------------------------------------------------
    # --- Figure set‑up -----------------------------------------------
    # ------------------------------------------------------------------
    ncols = n_targets + 1  # last col = overall panel
    fig, axes = plt.subplots(rows, ncols, figsize=(2.6 * ncols, 2.1 * rows), sharex="col")
    if rows == 1:
        axes = np.expand_dims(axes, 0)
    if ncols == 1:
        axes = np.expand_dims(axes, 1)

    trial_length_min = min(data_per_filter[fast_label]["trial_length"], data_per_filter[slow_label]["trial_length"])

    y_limits_per_row: list[tuple[int, tuple[float, float]]] = []

    # ==================================================================
    #  ROW LOOP ---------------------------------------------------------
    # ==================================================================
    for row_idx in range(rows):
        sig_masks_for_union = []

        # --------------------------------------------------------------
        #  Per‑target columns
        # --------------------------------------------------------------
        for col_idx, i_target in enumerate(range(n_targets)):
            ax = axes[row_idx, col_idx]
            colour = colour_palette.get(i_target, "#000000")
            means, data_A, data_B = [], [], []

            for f_idx, trf in enumerate(trial_filters):
                tgt_id = np.unique(data_per_filter[fast_label]["target_ids"])[i_target]
                mask = data_per_filter[trf]["target_ids"] == tgt_id
                data = data_per_filter[trf]["firing_rates"][mask, :trial_length_min, row_idx]

                mean_t = data.mean(0)
                sem_t = data.std(0, ddof=1) / np.sqrt(max(1, data.shape[0]))

                ls = "--" if f_idx == 0 else "-"
                ax.plot(mean_t, color=colour, linestyle=ls, linewidth=1.5)
                ax.fill_between(np.arange(mean_t.size), mean_t - sem_t, mean_t + sem_t, color=colour, alpha=0.20)
                means.append(mean_t)
                (data_A if f_idx == 0 else data_B).append(data)

                if target_tuned is not None and i_target == target_tuned[row_idx]:
                    for s in ax.spines.values():
                        s.set_edgecolor(colour); s.set_linewidth(2.5)

            mean_diff = np.abs(means[0] - means[1])
            ax.plot(mean_diff, color="black", linestyle="--", linewidth=1.5)

            if plot_significance:
                dA = np.concatenate(data_A, 0)  # (nA × time)
                dB = np.concatenate(data_B, 0)  # (nB × time)
                pvals = np.ones(trial_length_min)
                for t in range(trial_length_min):
                    a = dA[:, t]
                    b = dB[:, t]
                    a = a[~np.isnan(a)]
                    b = b[~np.isnan(b)]
                    pvals[t] = _p_value(a, b, use_paired=False)
                reject, *_ = multipletests(pvals, alpha=significance_alpha, method=correction_method)
                # reject = pvals < significance_alpha
                sig_masks_for_union.append(reject)
                blocks = _find_contiguous_true(reject)
                y0, y1 = ax.get_ylim(); yr = y1 - y0; y_bar = y1 + 0.05 * yr
                for s, e in blocks:
                    ax.hlines(y_bar, s, e, color="red", linewidth=1.2)
                    ax.text((s + e) / 2, y_bar + 0.02 * yr, "*", color="red", ha="center", va="bottom")
                ax.set_ylim(y0, y_bar + 0.08 * yr)

            if row_idx == 0:
                ax.set_title(f"Target {i_target + 1}")
            ax.set_xticks([])
            y_limits_per_row.append((row_idx, ax.get_ylim()))

        # --------------------------------------------------------------
        #  Overall column (mean‑difference + mean fast/slow)
        # --------------------------------------------------------------
        ax_mean = axes[row_idx, ncols - 1]
        diffs_per_tgt, fast_curves, slow_curves = [], [], []

        for i_target in range(n_targets):
            tgt_id = np.unique(data_per_filter[fast_label]["target_ids"])[i_target]
            m_f = data_per_filter[fast_label]["target_ids"] == tgt_id
            m_s = data_per_filter[slow_label]["target_ids"] == tgt_id
            d_f = data_per_filter[fast_label]["firing_rates"][m_f, :trial_length_min, row_idx]
            d_s = data_per_filter[slow_label]["firing_rates"][m_s, :trial_length_min, row_idx]
            fast_curves.append(d_f.mean(0))
            slow_curves.append(d_s.mean(0))
            diffs_per_tgt.append(np.abs(d_f.mean(0) - d_s.mean(0)))

        fast_curves = np.stack(fast_curves)
        slow_curves = np.stack(slow_curves)
        diffs_per_tgt = np.stack(diffs_per_tgt)

        # grand averages
        mean_fast = fast_curves.mean(0); sem_fast = fast_curves.std(0, ddof=1) / np.sqrt(n_targets)
        mean_slow = slow_curves.mean(0); sem_slow = slow_curves.std(0, ddof=1) / np.sqrt(n_targets)
        mean_diff = diffs_per_tgt.mean(0); sem_diff = diffs_per_tgt.std(0, ddof=1) / np.sqrt(n_targets)

        # plot
        ax_mean.plot(mean_fast, color='black', linestyle="--", linewidth=1.5, label="Fast avg")
        ax_mean.fill_between(np.arange(mean_fast.size), mean_fast - sem_fast, mean_fast + sem_fast, color='black', alpha=0.20)
        ax_mean.plot(mean_slow, color='black', linestyle="-", linewidth=1.5, label="Slow avg")
        ax_mean.fill_between(np.arange(mean_slow.size), mean_slow - sem_slow, mean_slow + sem_slow, color='black', alpha=0.20)
        ax_mean.plot(mean_diff, color="black", linestyle="--", linewidth=1.5, label="|diff|")
        ax_mean.fill_between(np.arange(mean_diff.size), mean_diff - sem_diff, mean_diff + sem_diff, color="black", alpha=0.25)
        ax_mean.set_title("All targets")
        ax_mean.set_xlabel("Time bins")

        if plot_significance and sig_masks_for_union:
            union_mask = np.any(np.vstack(sig_masks_for_union), axis=0)
            blocks = _find_contiguous_true(union_mask)
            y0, y1 = ax_mean.get_ylim(); yr = y1 - y0; y_bar = y1 + 0.05 * yr
            for s, e in blocks:
                ax_mean.hlines(y_bar, s, e, color="red", linewidth=1.2)
                ax_mean.text((s + e)/2, y_bar + 0.02 * yr, "*", color="red", ha="center", va="bottom")
            ax_mean.set_ylim(y0, y_bar + 0.08 * yr)

        if row_idx == 0:
            ax_mean.legend(frameon=False, fontsize=8, loc="upper right")

    # ------------------------------------------------------------------
    # --- Optional row‑wise y‑locking ---------------------------------
    # ------------------------------------------------------------------
    if lock_row_ylim:
        for r in range(rows):
            lims = [l for idx, l in y_limits_per_row if idx == r]
            mn = min(l[0] for l in lims); mx = max(l[1] for l in lims)
            for c in range(ncols):
                axes[r, c].set_ylim(mn, mx)

    for r in range(rows):
        axes[r, 0].set_ylabel(row_labels[r], rotation=0, labelpad=40)
    axes[-1, 0].set_xlabel("Time bins")
    plt.tight_layout()

    # ------------------------------------------------------------------
    # --- Save figure --------------------------------------------------
    # ------------------------------------------------------------------
    session_vis_dir = os.path.join(vis_dir, session_data_name)
    os.makedirs(session_vis_dir, exist_ok=True)

    suffix = '_neurons'
    suffix += "_sig" if plot_significance else ""
    suffix += f"_rows{rows}"
    suffix += f"_max{max_neurons_plot}" if max_neurons_plot else ""
    suffix += f"_vd{visual_delay_time}" if visual_delay_time else ""
    fname = f"{session_data_name}_direction_tuned{suffix}.pdf"
    plt.savefig(os.path.join(session_vis_dir, fname), dpi=600)
    plt.close()


def plot_neurons_by_target_single_session(
    session_data_name: str,
    unit_filter: str,
    window_config: str,
    time_step: float,
    data_format: str,
    trial_filters: tuple[str, str],
    neuron_indices: list[int],
    *,
    truncate_percentile: int = None,
    neuron_colors: list[str] | None = None,
    visual_delay_time: float | None = 0.1,
    peak_time: float = 0.5,
    significance_alpha: float = 0.05,
    correction_method: str = "fdr_bh",
):
    """Generate **nine square pdfs**: one for each of 8 targets + 1 grand‑average.

    Each pdf overlays 2‑5 selected neurons in different colours; for every
    neuron we draw:
        • fast condition → dashed line (colour *n*)
        • slow condition → solid line  (colour *n*)
        • optional ± SEM shading (20 % opacity)

    The x‑ and y‑axes are locked across all nine plots, so the files can be
    tiled without visual jumps.

    Parameters
    ----------
    session_data_name : str
        Folder / identifier for the session.
    unit_filter : str
        Which neuronal subset to load (e.g. 'MC').
    neuron_indices : list‑like of int
        2–5 neuron indices to visualise.
    colour_map : dict[int, str] | None
        Map neuron‑index → colour.  If *None*, a default matplotlib cycle is
        used in the given order of *neuron_indices*.
    visual_delay_time : float | None
        Optional time‑shift to align traces with behavioural markers.
    data_format : str, default 'truncate_end'
        Must match what your DataLoader expects.
    significance_alpha / correction_method
        Same semantics as in the dense plot: if you want time‑bin significance
        stars, simply re‑use those values; set *significance_alpha=None* to
        skip stats.
    """

    ## Sanity & colour setup
    assert len(neuron_indices) <= 5, "Function expects less than 5 neurons."
    assert data_format == "truncate_end", "Only 'truncate_end' supported."

    if neuron_colors is None:
        default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colour_map = {idx: default_cycle[i % len(default_cycle)] for i, idx in enumerate(neuron_indices)}
    else:
        assert len(neuron_indices) == len(neuron_colors), "Mismatch between neuron indices and colours."
        colour_map = {idx: col for idx, col in zip(neuron_indices, neuron_colors)}

    fast_label, slow_label = trial_filters
    fast_style = {"linestyle": "--"}  # colour supplied on call
    slow_style = {"linestyle": "-"}

    ## Load data
    # vis_shift = int(visual_delay_time / config.time_step) if visual_delay_time else 0
    data_per: dict[str, dict] = {}
    for tf in trial_filters:
        dl = utils_processing.DataLoader(
            data_dir=data_dir,
            results_dir=results_dir,
            session_data_name=session_data_name,
            unit_filter=unit_filter,
            input_unit_filter=None,
            window_config=window_config,
            trial_filter=tf,
        )
        dl.load_firing_rate_data()
        dl.load_cursor_data()
        dl.remove_target_overlap(target_radius=session_target_radii[session_data_name])

        t_ids = dl.get_target_ids()
        
        fr, *_ , keep = dl.reformat_firing_rate_data(data_format=data_format, trial_length_filter_percentile=truncate_percentile)
        t_ids = t_ids[keep]
        # if vis_shift:
        #     fr = fr[:, vis_shift:, :]
        fr = fr[:, :, neuron_indices]
        data_per[tf] = {"target_ids": t_ids, "fr": fr, "T": fr.shape[1]}

    n_targets = len(np.unique(data_per[fast_label]["target_ids"]))
    T_min = min(data_per[fast_label]["T"], data_per[slow_label]["T"])
    time_axis = np.arange(T_min) * time_step  # seconds

    ## Mean / SEM pre‑calc
    mean_dct, sem_dct = {}, {}
    y_min, y_max = np.inf, -np.inf

    ## per‑target stats
    for tgt in range(n_targets):
        tgt_id = np.unique(data_per[fast_label]["target_ids"])[tgt]
        for tf in trial_filters:
            # ipdb.set_trace()

            mask = data_per[tf]["target_ids"] == tgt_id
            fr = data_per[tf]["fr"][mask, :T_min, :]  # trials × time × neurons
            mean = fr.mean(0)
            sem  = fr.std(0, ddof=1) / np.sqrt(max(1, fr.shape[0]))
            mean_dct[(tgt, tf)] = mean
            sem_dct[(tgt, tf)]  = sem
            y_min = min(y_min, np.min(mean - sem))
            y_max = max(y_max, np.max(mean + sem))

    ## pooled (ALL) stats — use **all trials**, not the average of targets
    for tf in trial_filters:
        fr_all = data_per[tf]["fr"][:, :T_min, :]  # all trials × time × neurons
        mean_all = fr_all.mean(0)
        sem_all  = fr_all.std(0, ddof=1) / np.sqrt(fr_all.shape[0])
        mean_dct[("all", tf)] = mean_all
        sem_dct[("all", tf)]  = sem_all
        y_min = min(y_min, np.min(mean_all - sem_all))
        y_max = max(y_max, np.max(mean_all + sem_all))

    ## Significance masks (target panels only)
    sig_masks = {}
    if significance_alpha is not None:
        for tgt in range(n_targets):
            tgt_id = np.unique(data_per[fast_label]["target_ids"])[tgt]
            for n_idx, neuron in enumerate(neuron_indices):
                pvals = np.ones(T_min)
                mask_f = data_per[fast_label]["target_ids"] == tgt_id
                mask_s = data_per[slow_label]["target_ids"] == tgt_id
                A = data_per[fast_label]["fr"][mask_f, :T_min, n_idx]
                B = data_per[slow_label]["fr"][mask_s, :T_min, n_idx]
                for t in range(T_min):
                    a = A[:, t].ravel(); b = B[:, t].ravel()
                    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
                    pvals[t] = _p_value(a, b, use_paired=False)
                sig_masks[(tgt, neuron)], *_ = multipletests(pvals, alpha=significance_alpha, method=correction_method)

    ## Plot helper
    out_dir = os.path.join(vis_dir, session_data_name)
    os.makedirs(out_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────
    # NEW block that **replaces** your nine-PDF loop at the end
    # ──────────────────────────────────────────────────────────────────────
    # Mapping of target index → (row, col) in a 3×3 grid (centre is empty)
    idx2pos = {0: (1, 2),  # right
               1: (0, 2),  # upper-right
               2: (0, 1),  # up
               3: (0, 0),  # upper-left
               4: (1, 0),  # left
               5: (2, 0),  # lower-left
               6: (2, 1),  # down
               7: (2, 2)}  # lower-right

    fig, ax_grid = plt.subplots(3, 3, figsize=(45*mm, 45*mm),
                                sharex=True, sharey=True)
    ax_grid = ax_grid.reshape(3, 3)

    # Draw every target around the “ring”
    for tgt_idx in range(n_targets):          # 0‥7
        r, c = idx2pos[tgt_idx]
        tgt_colour = target_color_palette_8[tgt_idx]
        _plot_target_panel(ax_grid[r, c], tgt_idx,
                           time_axis=time_axis,
                           mean_dct=mean_dct, sem_dct=sem_dct,
                           sig_masks=sig_masks,
                           neuron_indices=neuron_indices,
                           colour_map=colour_map,
                           target_colour=tgt_colour,
                           fast_label=fast_label, slow_label=slow_label,
                           y_min=y_min, y_max=y_max,
                           visual_delay_time=visual_delay_time,
                           peak_time=peak_time,
                           significance_alpha=significance_alpha)

    # remove the centre axis; optionally put the grand-average there
    ax_grid[1, 1].set_visible(False)

    # (optional) show global axis labels once, at bottom centre / middle left
    # ax_grid[2, 0].set_xlabel("Time (s)", fontsize=5)
    # ax_grid[2, 0].set_ylabel("Firing rate (Hz)", fontsize=5)
    
    label_ax = ax_grid[2, 0]           # lower-left keeps the labels
    for r in range(3):
        for c in range(3):
            ax = ax_grid[r, c]
            if ax is label_ax or not ax.get_visible():
                continue               # leave the master axis alone
            ax.tick_params(axis="both", which="both",
                           labelbottom=False, labelleft=False,
                           bottom=False, left=False)   # hides only on this axes

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{session_data_name}_targets_ring.pdf"),
                dpi=600, bbox_inches=None, transparent=True, format="pdf")
    plt.close(fig)


def _plot_target_panel(ax, tgt_key, *,
                       time_axis, mean_dct, sem_dct,
                       sig_masks, neuron_indices, colour_map,
                       target_colour=None,
                       fast_label, slow_label,
                       y_min, y_max,
                       visual_delay_time,
                       peak_time,
                       significance_alpha):
    """Draw one target into the supplied `ax` (no figure creation here!)."""
    fast_style = {"linestyle": "--"}
    slow_style = {"linestyle": "-"}

    for n_idx, neuron in enumerate(neuron_indices):
        col = target_colour if target_colour is not None else colour_map[neuron]

        # fast & slow means / SEM
        m_fast = mean_dct[(tgt_key, fast_label)][:, n_idx]
        s_fast = sem_dct[(tgt_key, fast_label)][:, n_idx]
        m_slow = mean_dct[(tgt_key, slow_label)][:, n_idx]
        s_slow = sem_dct[(tgt_key, slow_label)][:, n_idx]

        ax.plot(time_axis, m_fast, **fast_style, color=col, linewidth=size_line_thin, alpha=alpha_line)
        ax.fill_between(time_axis, m_fast-s_fast, m_fast+s_fast, color=col, alpha=0.15)

        ax.plot(time_axis, m_slow, **slow_style, color=col, linewidth=size_line_thin, alpha=alpha_line)
        ax.fill_between(time_axis, m_slow-s_slow, m_slow+s_slow, color=col, alpha=0.25)

        # optional significance band
        if significance_alpha is not None:
            mask   = sig_masks[(tgt_key, neuron)]
            blocks = _find_contiguous_true(mask)
            for s, e in blocks:
                ax.hlines(y_max, time_axis[s], time_axis[e],
                          color='k', linewidth=size_line_thin, alpha=alpha_line)
                ax.text((time_axis[s] + time_axis[e]) / 2, y_max - 0.75,
                        "*", color='k', ha="center", va="bottom", fontsize=7)

    # time-zero marker
    ax.axvline(visual_delay_time, color="black", linewidth=size_line_thin,
               linestyle="--", alpha=alpha_line)
    ax.axvline(peak_time, color="black", linewidth=size_line_thin,
               linestyle=":", alpha=alpha_line)

    # global limits and cosmetics
    ax.set_xlim(time_axis[0], time_axis[-1])
    ax.set_ylim(y_min, y_max + 0.15 * (y_max - y_min))
    ax.set_aspect((time_axis[-1] - time_axis[0]) /
                  (y_max - y_min + 0.15 * (y_max - y_min)))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_percent_neuron_discrepancy(
    session_data_names: list[str],
    unit_filter: str,
    time_step: float,
    window_config: str,
    data_format: str,
    trial_filters: tuple[str, str],
    reaction_times: list[float],
    visual_delay_times: list[float],
    peak_times: list[float],
    *,
    truncate_percentile: int | None = None,
    significance_alpha: float = 0.05,
    correction_method: str = "fdr_bh",
    pre_start_time_buffer: float = 0.0,
    post_reach_time_buffer: float = 0.0,
    save_fig: bool = True,
):
    """
    As before, compute the % of neurons whose response differs between *fast* and *slow* trials in three epochs
    (visual delay, before‑peak, after‑peak) for each target and an "ALL" summary column.  Visualisation has been
    switched to **swarm plots**:
        • Every session = one dot → full transparency on between‑session spread.
        • Central tendency = median line per phase.
        • Whiskers = min–max (so significance bands still have a clear reference height).
    """

    # --- (1)  DATA COLLECTION  --------------------------------------------------------------------------
    assert len(reaction_times)     == len(session_data_names)
    assert len(visual_delay_times) == len(session_data_names)
    assert len(peak_times)         == len(session_data_names)

    reaction_indices     = (np.array(reaction_times) / time_step).astype(int)
    visual_delay_indices = (np.array(visual_delay_times) / time_step).astype(int)
    peak_indices         = (np.array(peak_times) / time_step).astype(int)
    pre_start_idx        = int(pre_start_time_buffer / time_step)
    post_reach_idx       = int(post_reach_time_buffer / time_step)

    fast_label, slow_label = trial_filters

    percent_visual_delay_all = []
    percent_before_peak_all  = []
    percent_after_peak_all   = []

    # task name inferred from first session for colour palette lookup
    task_name = session_data_names[0].split('_')[-1]

    for session_data_name, reaction_idx, visual_delay_idx, peak_idx in zip(
        session_data_names, reaction_indices, visual_delay_indices, peak_indices):

        task_name_ = session_data_name.split('_')[-1]
        assert task_name_ == task_name, f"All session names must have the same task suffix, got {task_name_} vs {task_name}."

        # peak_onset_time = config.session_data_dict[session_data_name]['peak_onset_time']['all']
        # peak_time = config.session_data_dict[session_data_name]['peak_time']['all']

        # if unit_filter == 'MC-LAT':
        #     peak_onset_time = peak_onset_time[0]
        #     peak_time = peak_time[0]
        # elif unit_filter == 'MC-MED':
        #     peak_onset_time = peak_onset_time[1]
        #     peak_time = peak_time[1]

        # visual_delay_idx = int(peak_onset_time / config.time_step)
        # peak_idx = int(peak_time / config.time_step)

        data_per_filter: dict[str, dict] = {}

        for trial_filter in trial_filters:
            dl = utils_processing.DataLoader(
                data_dir=data_dir,
                results_dir=results_dir,
                session_data_name=session_data_name,
                unit_filter=unit_filter,
                input_unit_filter=None,
                window_config=window_config,
                trial_filter=trial_filter,
            )
            dl.load_firing_rate_data()
            dl.load_cursor_data()
            dl.remove_target_overlap(target_radius=session_target_radii[session_data_name])

            target_ids = dl.get_target_ids()

            percentile = (
                truncate_percentile
                if truncate_percentile is not None
                else None
            )
            firing_rates_simple, _, _, _, _, keep_mask = dl.reformat_firing_rate_data(
                data_format=data_format, trial_length_filter_percentile=percentile
            )
            if reaction_idx > 0:
                firing_rates_simple = firing_rates_simple[:, reaction_idx:, :]

            target_ids = target_ids[keep_mask]  # keep-mask returned as last item

            data_per_filter[trial_filter] = {
                "target_ids": target_ids,
                "firing_rates": firing_rates_simple,  # (trials, T, neurons)
                "trial_length": firing_rates_simple.shape[1],
                "keep_mask": keep_mask,
            }

        # Compute adjusted epoch indices once (accounting for reaction trimming)
        adj_pre_start_idx  = max(0, pre_start_idx - reaction_idx)
        adj_visual_delay_idx = visual_delay_idx
        adj_peak_idx         = peak_idx

        # harmonise trial length across conditions
        T_min = min(data_per_filter[fast_label]["trial_length"],
                    data_per_filter[slow_label]["trial_length"])

        # Movement-epoch boundaries within the data array
        movement_start = adj_pre_start_idx
        movement_end   = T_min - post_reach_idx if post_reach_idx > 0 else T_min
        if movement_end <= movement_start:
            print(f'  Warning: post_reach_idx={post_reach_idx} leaves no movement bins '
                  f'(T_min={T_min}, movement_start={movement_start}) for {session_data_name}. '
                  f'Using T_min as movement_end.')
            movement_end = T_min
        vd_end = min(movement_start + adj_visual_delay_idx, movement_end)
        pk_end = min(movement_start + adj_peak_idx, movement_end)

        n_neurons = data_per_filter[fast_label]["firing_rates"].shape[-1]
        n_targets = len(np.unique(data_per_filter[fast_label]["target_ids"]))

        # Significance test every neuron, every target
        # arrays shape (n_targets, n_neurons, 3) → 3 for visual delay phase, before and after peak
        sig_matrix = np.zeros((n_targets, n_neurons, 3), dtype=bool)

        for tgt_i in range(n_targets):
            tgt_label = np.unique(data_per_filter[fast_label]["target_ids"])[tgt_i]

            # masks for this target
            m_fast = data_per_filter[fast_label]["target_ids"] == tgt_label
            if 'far' in trial_filters:
                m_slow = data_per_filter[slow_label]["target_ids"] == (tgt_label + n_targets)
            else:
                m_slow = data_per_filter[slow_label]["target_ids"] == tgt_label

            fast_trials = data_per_filter[fast_label]["firing_rates"][m_fast, :T_min, :]
            slow_trials = data_per_filter[slow_label]["firing_rates"][m_slow, :T_min, :]

            for n_idx in range(n_neurons):
                A = fast_trials[:, :, n_idx]
                B = slow_trials[:, :, n_idx]

                # FDR applied across movement bins only (visual delay → trial completion)
                pvals_movement = np.array([
                    _p_value(A[:, t], B[:, t], use_paired=False)
                    for t in range(movement_start, movement_end)
                ])
                reject_movement, *_ = multipletests(
                    pvals_movement, alpha=significance_alpha, method=correction_method
                )

                # Re-embed into full array for clean epoch indexing
                reject = np.zeros(T_min, dtype=bool)
                reject[movement_start:movement_end] = reject_movement

                sig_matrix[tgt_i, n_idx, 0] = reject[movement_start:vd_end].any()
                sig_matrix[tgt_i, n_idx, 1] = reject[vd_end:pk_end].any()
                sig_matrix[tgt_i, n_idx, 2] = reject[pk_end:movement_end].any()


        # percentages per target
        percent_visual_delay = sig_matrix[:, :, 0].sum(axis=1) / n_neurons * 100
        percent_before_peak  = sig_matrix[:, :, 1].sum(axis=1) / n_neurons * 100
        percent_after_peak   = sig_matrix[:, :, 2].sum(axis=1) / n_neurons * 100
        
        # “overall” column (any target significant counts)
        sig_union = sig_matrix.any(axis=0)  # shape (n_neurons, 2)
        overall_visual_delay = sig_union[:, 0].sum() / n_neurons * 100
        overall_before_peak  = sig_union[:, 1].sum() / n_neurons * 100
        overall_after_peak   = sig_union[:, 2].sum() / n_neurons * 100

        # concatenate for plotting
        percent_visual_delay = np.append(percent_visual_delay, overall_visual_delay)
        percent_before_peak  = np.append(percent_before_peak, overall_before_peak)
        percent_after_peak   = np.append(percent_after_peak, overall_after_peak)

        percent_visual_delay_all.append(percent_visual_delay)
        percent_before_peak_all.append(percent_before_peak)
        percent_after_peak_all.append(percent_after_peak)

    # (stack session × target/ALL) shape → (n_sessions, n_groups)
    percent_visual_delay_all = np.stack(percent_visual_delay_all)
    percent_before_peak_all  = np.stack(percent_before_peak_all)
    percent_after_peak_all   = np.stack(percent_after_peak_all)

    # --- Save numerical results for reuse (e.g. by _all_sessions plot) ---
    results_cache = {
        "percent_visual_delay_all": percent_visual_delay_all,
        "percent_before_peak_all": percent_before_peak_all,
        "percent_after_peak_all": percent_after_peak_all,
        "session_data_names": session_data_names,
        "unit_filter": unit_filter,
        "trial_filters": trial_filters,
        "task_name": task_name,
    }
    cache_fname = (
        f"{task_name}_{len(session_data_names)}_sessions_"
        f"percent_neuron_discrepancy_{unit_filter}_{data_format}.pkl"
    )
    with open(os.path.join(results_dir, cache_fname), "wb") as fh:
        pickle.dump(results_cache, fh)

    # Prep DataFrame for seaborn
    labels = [f"T{t+1}" for t in range(percent_visual_delay_all.shape[1] - 1)] + ["ALL"]

    phase_names  = ["Visual delay", "Before peak", "After peak"]
    phase_arrays = [percent_visual_delay_all, percent_before_peak_all, percent_after_peak_all]

    # Build long‑form table: one row per (session, target, phase)
    rows = []
    for phase_name, arr in zip(phase_names, phase_arrays):
        for sess_i in range(arr.shape[0]):
            for tgt_i, tgt_label in enumerate(labels):
                rows.append((tgt_label, phase_name, float(arr[sess_i, tgt_i])))

    df = pd.DataFrame(rows, columns=["Target", "Phase", "Percent"])

    #Figure & swarmplot

    palette = {
        phase_names[0]: color_palettes[task_name][trial_filters[0]][2],
        phase_names[1]: color_palettes[task_name][trial_filters[0]][1],
        phase_names[2]: color_palettes[task_name][trial_filters[0]][0],
    }

    fig, ax = plt.subplots(figsize=(88 * mm, 40 * mm))
    sns.swarmplot(
        data=df,
        x="Target", y="Percent",
        hue="Phase",
        dodge=True,
        palette=palette,
        # linewidth=0.25,
        # edgecolor="k",
        size=3,
        ax=ax,
        # zorder=3,
        rasterized=False,
        alpha=0.8,
        legend=False,
    )

    # Significance lines over “ALL” column
    # def _star(p):
    #     return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "NS"

    # i_all = len(labels) - 1
    # p_vd_vs_bp = wilcoxon(percent_visual_delay_all[:, i_all], percent_before_peak_all[:, i_all]).pvalue
    # p_vd_vs_ap = wilcoxon(percent_visual_delay_all[:, i_all], percent_after_peak_all[:, i_all]).pvalue
    # p_bp_vs_ap = wilcoxon(percent_before_peak_all[:, i_all], percent_after_peak_all[:, i_all]).pvalue

    # stars = {"vd-bp": _star(p_vd_vs_bp),
    #          "vd-ap": _star(p_vd_vs_ap),
    #          "bp-ap": _star(p_bp_vs_ap)}

    # Max y per phase “ALL” column → anchor for bars
    # top_vd = percent_visual_delay_all[:, i_all].max()
    # top_bp = percent_before_peak_all[:, i_all].max()
    # top_ap = percent_after_peak_all[:, i_all].max()

    # Compute x positions like seaborn’s dodge: centres ± width/2
    # group_centres = np.arange(len(labels))
    # dodge_shift   = 0.20  # empirically matches seaborn default for 3 hues
    # x_vd = group_centres[i_all] - dodge_shift
    # x_bp = group_centres[i_all]           # middle
    # x_ap = group_centres[i_all] + dodge_shift

    # def _band(x_left, x_right, text, y):
    #     ax.plot([x_left, x_right], [y, y], color="k", linewidth=0.7)
    #     ax.text((x_left + x_right) / 2, y + 0.4, text, ha="center", va="bottom", fontsize=6)

    # y_base = max(top_vd, top_bp, top_ap) + 2.0
    # _band(x_vd, x_bp, stars["vd-bp"], y_base)
    # _band(x_vd, x_ap, stars["vd-ap"], y_base + 4)
    # _band(x_bp, x_ap, stars["bp-ap"], y_base + 8)

    # Legend – custom handles (dot only) with adjustable edge width
    # handles = [Line2D([0], [0], marker="o", markersize=3, markerfacecolor=palette[pn],
    #                   markeredgecolor="k", markeredgewidth=0, label=pn,
    #                   linestyle="none")
    #            for pn in phase_names]
    # legend = ax.legend(handles=handles, fontsize=5, loc="upper left")
    # legend.get_frame().set_alpha(1.0)

    # Aesthetics
    # set x labels to nothing
    ax.set_xlabel("")
    ax.set_ylabel("% Neurons with cross-condition discrepancy", fontsize=5)
    ax.set_ylim(-2, 50)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5, linewidth=0.25)
    plt.tight_layout()

    # Save or show
    if save_fig:
        fname = f"{task_name}_{len(session_data_names)}_sessions_percent_neuron_discrepancy_swarm_seaborn_{unit_filter}_{data_format}.pdf"
        plt.savefig(os.path.join(vis_dir, fname), bbox_inches=None, format='pdf', dpi=600)
        plt.close()
    else:
        plt.show()


def _neuron_target_tuned_one_session(
        session_data_name: str,
        unit_filter: str,
        time_step: float,
        window_config: str,
        data_format: str,
        trial_filters: tuple[str, str],
        pre_start_time: float,
        *,
        truncate_percentile: int | None,
        significance_alpha: float,
):
    """
    Determine which neurons are *tuned* to at least one target.

    A neuron is considered tuned (for a given target/condition) if its
    firing-rate distribution after target onset differs significantly from
    the baseline distribution that immediately precedes the start cue.

    Returns
    -------
    tuned_matrix : (n_targets, n_neurons, 2) bool
        Dimension-2 indices 0/1 correspond to `trial_filters[0]`
        (e.g. *fast*) and `trial_filters[1]` (e.g. *slow*).
    tuned_union  : (n_neurons,) bool
        True if the neuron is tuned to **any** target in **either**
        condition.
    """
    # ---------------------------------------------------------------------
    pre_start_index = int(pre_start_time / time_step)
    assert pre_start_index >= 0, "`pre_start_time` must be ≥ 0"

    fast_label, slow_label = trial_filters
    data_per_filter: dict[str, dict] = {}

    # -------------------- load & split baseline / active -----------------
    for trial_filter in trial_filters:
        dl = utils_processing.DataLoader(
            data_dir=data_dir,
            results_dir=results_dir,
            session_data_name=session_data_name,
            unit_filter=unit_filter,
            input_unit_filter=None,
            window_config=window_config,
            trial_filter=trial_filter,
        )
        dl.load_firing_rate_data()
        dl.load_cursor_data()
        dl.remove_target_overlap(
            target_radius=session_target_radii[session_data_name]
        )

        tgt_ids = dl.get_target_ids()
        fr, _, _, _, _, keep = dl.reformat_firing_rate_data(
            data_format=data_format,
            trial_length_filter_percentile=truncate_percentile,
        )

        # Split into baseline (pre-start) and active (post-start)
        baseline_fr = fr[:, :pre_start_index, :]          # (trials, T₀, neurons)
        active_fr   = fr[:, pre_start_index:, :]          # (trials, T₁, neurons)

        tgt_ids = tgt_ids[keep]                           # drop trials filtered out
        data_per_filter[trial_filter] = {
            "target_ids": tgt_ids,
            "baseline_fr": baseline_fr,
            "active_fr": active_fr,
        }

    # ------------------------- book-keeping ------------------------------
    unique_targets = np.unique(data_per_filter[fast_label]["target_ids"])
    n_targets      = len(unique_targets)
    n_neurons      = data_per_filter[fast_label]["baseline_fr"].shape[-1]

    # boolean tuned/not-tuned for each target × neuron × condition
    tuned_matrix = np.zeros((n_targets, n_neurons, 2), dtype=bool)

    # -------------------- per-target statistical test --------------------
    for tgt_idx, tgt_val in enumerate(unique_targets):
        for cond_idx, trial_filter in enumerate(trial_filters):
            ids       = data_per_filter[trial_filter]["target_ids"]
            baseline  = data_per_filter[trial_filter]["baseline_fr"][ids == tgt_val]
            active    = data_per_filter[trial_filter]["active_fr"][ids == tgt_val]

            # If no trials for this target/condition, skip
            if baseline.size == 0 or active.size == 0:
                continue

            # Stack *all* time samples from all trials to boost power
            #   → shape (samples, neurons)
            base_flat = baseline.reshape(-1, n_neurons)
            act_flat  = active.reshape(-1, n_neurons)

            for n in range(n_neurons):
                p = _p_value(base_flat[:, n], act_flat[:, n], use_paired=False)
                tuned_matrix[tgt_idx, n, cond_idx] = p < significance_alpha

    # ---------------------- neuron-wise union mask -----------------------
    # tuned_union = tuned_matrix.any(axis=(0, 2))   # collapse targets & conditions
    return tuned_matrix


def _neuron_cross_condition_discrepancy_one_session(
        session_data_name: str,
        unit_filter: str,
        time_step: float,
        window_config: str,
        data_format: str,
        trial_filters: tuple[str, str],
        reaction_time: float,
        visual_delay_time: float,
        peak_time: float,
        *,
        truncate_percentile: int | None,
        significance_alpha: float,
        correction_method: str,
        pre_start_time_buffer: float = 0.0,
        post_reach_time_buffer: float = 0.0,
        separate_correction_per_window=False):
    """
    Returns a (n_targets, n_neurons, 3) bool array indicating whether each
    neuron shows a significant firing-rate difference between conditions in
    each of the three movement epochs: visual delay, before-peak, after-peak.

    Pre-movement and post-movement buffer bins are sliced off each trial
    individually before aligning trial lengths via truncate_end.  This
    guarantees that truncation is based on movement duration only and that
    the FDR test covers exactly the movement-relevant bins.
    """
    r_idx          = int(reaction_time         / time_step)
    vd_idx         = int(visual_delay_time     / time_step)
    pk_idx         = int(peak_time             / time_step)
    pre_start_idx  = int(pre_start_time_buffer  / time_step)
    post_reach_idx = int(post_reach_time_buffer / time_step)
    move_start_idx = pre_start_idx + r_idx   # first movement bin in raw array

    fast_label, slow_label = trial_filters
    data_per_filter: dict[str, dict] = {}

    for trial_filter in trial_filters:
        dl = utils_processing.DataLoader(
            data_dir=data_dir,
            results_dir=results_dir,
            session_data_name=session_data_name,
            unit_filter=unit_filter,
            input_unit_filter=None,
            window_config=window_config,
            trial_filter=trial_filter,
        )
        dl.load_firing_rate_data()
        dl.load_cursor_data()
        dl.remove_target_overlap(target_radius=session_target_radii[session_data_name])

        tgt_ids = dl.get_target_ids()

        # Load raw variable-length trials (no truncation) with percentile filtering
        fr_raw, _, _, _, _, keep = dl.reformat_firing_rate_data(
            data_format=None, trial_length_filter_percentile=truncate_percentile)
        tgt_ids = tgt_ids[keep]

        # Slice pre-movement and post-movement buffer from each trial,
        # then align to minimum movement duration via truncate_end.
        movement_slices = []
        valid = np.ones(len(fr_raw), dtype=bool)
        for i, fr_trial in enumerate(fr_raw):
            T_i = fr_trial.shape[0]
            end = T_i - post_reach_idx if post_reach_idx > 0 else T_i
            if end <= move_start_idx:
                valid[i] = False
            else:
                movement_slices.append(fr_trial[move_start_idx:end])
        tgt_ids = tgt_ids[valid]

        fr, _, _ = utils_processing.truncate_emissions(
            movement_slices, truncate_end=True)

        data_per_filter[trial_filter] = {
            "target_ids": tgt_ids,
            "firing_rates": fr,       # (n_trials, T_movement, N)
            "trial_length": fr.shape[1],
        }

    # T_min is now the minimum movement duration across conditions
    T_min = min(data_per_filter[fast_label]["trial_length"],
                data_per_filter[slow_label]["trial_length"])

    # Epoch boundaries are relative to movement onset (bin 0 = movement onset)
    vd_end = min(vd_idx, T_min)
    pk_end = min(pk_idx, T_min)

    n_neurons = data_per_filter[fast_label]["firing_rates"].shape[-1]
    n_targets = len(np.unique(data_per_filter[fast_label]["target_ids"]))
    sig_matrix = np.zeros((n_targets, n_neurons, 3), dtype=bool)

    for tgt_i, tgt_val in enumerate(np.unique(data_per_filter[fast_label]["target_ids"])):
        m_fast = data_per_filter[fast_label]["target_ids"] == tgt_val
        if 'far' in trial_filters:
            m_slow = data_per_filter[slow_label]["target_ids"] == (tgt_val + n_targets)
        else:
            m_slow = data_per_filter[slow_label]["target_ids"] == tgt_val

        fast_trials = data_per_filter[fast_label]["firing_rates"][m_fast, :T_min, :]
        slow_trials = data_per_filter[slow_label]["firing_rates"][m_slow, :T_min, :]

        if separate_correction_per_window:
            for n in range(n_neurons):
                A = fast_trials[:, :, n]
                B = slow_trials[:, :, n]

                for epoch_slice, epoch_i in [
                    (slice(0,      vd_end), 0),
                    (slice(vd_end, pk_end), 1),
                    (slice(pk_end, T_min),  2),
                ]:
                    pvals_epoch = np.array([
                        _p_value(A[:, t], B[:, t], use_paired=False)
                        for t in range(*epoch_slice.indices(T_min))
                    ])
                    if len(pvals_epoch) > 0:
                        reject_epoch, *_ = multipletests(
                            pvals_epoch, alpha=significance_alpha,
                            method=correction_method)
                        sig_matrix[tgt_i, n, epoch_i] = reject_epoch.any()

        else:
            for n in range(n_neurons):
                A = fast_trials[:, :, n]
                B = slow_trials[:, :, n]

                # FDR across all movement bins (visual delay → trial completion)
                pvals = np.array([
                    _p_value(A[:, t], B[:, t], use_paired=False)
                    for t in range(T_min)
                ])
                reject, *_ = multipletests(
                    pvals, alpha=significance_alpha, method=correction_method)

                sig_matrix[tgt_i, n, 0] = reject[:vd_end].any()
                sig_matrix[tgt_i, n, 1] = reject[vd_end:pk_end].any()
                sig_matrix[tgt_i, n, 2] = reject[pk_end:].any()

    return sig_matrix
    
    # # percent_neurons_by_target = sig_matrix.sum(axis=1) / n_neurons * 100.0
    # percent_neurons_by_target = sig_matrix.sum(axis=1)

    # # ALL column (union across targets)
    # sig_union       = sig_matrix.any(axis=0)          # (n_neurons, 3)
    # percent_neurons = sig_union.sum(axis=0) / n_neurons * 100.0


    # # ---------- NEW: restrict analysis to “responsive” neurons ----------
    # # “Responsive” = shows a significant effect in *any* target × epoch
    # # Step 1 – collapse across targets (→ n_neurons × 3)
    # # sig_union = sig_matrix.any(axis=0)

    # # # Step 2 – collapse across the 3 epochs to get a 1-D mask
    # # responsive_neurons = sig_union.any(axis=1)          # shape (n_neurons,)

    # # # Nothing to keep?  → fall back to old behaviour to avoid /0
    # # if responsive_neurons.any():
    # #     sig_matrix = sig_matrix[:, responsive_neurons, :]
    # # # --------------------------------------------------------------------

    # # n_resp = sig_matrix.shape[1]                        # after filtering
    # # percent_neurons_by_target = sig_matrix.sum(axis=1) / n_resp * 100.0

    # # sig_union       = sig_matrix.any(axis=0)            # (n_resp, 3)
    # # percent_neurons = sig_union.sum(axis=0) / n_resp * 100.0

    # return percent_neurons_by_target, percent_neurons


# ------------- helpers ---------------------------------------------------------
phase_names = ["Visual delay", "Before peak", "After peak"]

def _star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "NS"

# ==============================================================================
#  MULTI‑TASK BOX PLOT  (group‑specific palettes, no legend)
# ==============================================================================

def plot_percent_neuron_discrepancy_all_sessions(
    session_info_dict: dict[str, dict],
    time_step: float,
    window_config: str,
    data_format: str,
    reaction_time: float,
    *,
    truncate_percentile: int | None = None,
    significance_alpha: float = 0.05,
    correction_method: str = "fdr_bh",
    save_fig: bool = True,
    show_significance: bool = False,
):
    """Two figures from cached results:

    1. Box-plots (per-target values pooled across sessions) — main figure.
    2. Swarm plot of session averages — supplement figure.
    """

    subject_display = {"N1": "JJ", "N2": "RD"}

    # ------------------------------------------------------------------
    group_labels, group_palettes = [], []
    phase_arrays_all_groups, pvals_groups = [], []
    session_avgs_all_groups = []

    # ---- iterate groups ------------------------------------------------
    for info in session_info_dict.values():
        subj    = info["subject"]
        subj_display = subject_display.get(subj, subj)
        arrays  = info["unit_filters"]
        unit_filters_short = info["unit_filters_short"]
        task_short = info["task_short"]
        task_name  = info["task"]
        sess_names = info["session_data_names"]
        trial_filters = tuple(info["trial_filters"])

        for u_idx, array in enumerate(arrays):
            grp_lbl = f"{subj_display} {unit_filters_short[u_idx]} {task_short}"
            group_labels.append(grp_lbl)

            # palette (reversed: dark, mid, light)
            palette_3_orig = color_palettes[task_name][trial_filters[0]]
            palette_3 = [palette_3_orig[2], palette_3_orig[1], palette_3_orig[0]]
            group_palettes.append(palette_3)

            # --- Load cached results ------------------------------------
            cache_fname = (
                f"{task_name}_{len(sess_names)}_sessions_"
                f"percent_neuron_discrepancy_{array}_{data_format}.pkl"
            )
            cache_path = os.path.join(results_dir, cache_fname)
            with open(cache_path, "rb") as fh:
                cache = pickle.load(fh)

            pct_vd = cache["percent_visual_delay_all"]  # (n_sessions, n_targets+1)
            pct_bp = cache["percent_before_peak_all"]
            pct_ap = cache["percent_after_peak_all"]

            # Pool per-target values across sessions
            vis = pct_vd[:, :-1].flatten()
            bef = pct_bp[:, :-1].flatten()
            aft = pct_ap[:, :-1].flatten()

            phase_arrays_all_groups.append([vis, bef, aft])

            # Session averages (mean across targets per session)
            sess_avg_vis = pct_vd[:, :-1].mean(axis=1)
            sess_avg_bef = pct_bp[:, :-1].mean(axis=1)
            sess_avg_aft = pct_ap[:, :-1].mean(axis=1)
            session_avgs_all_groups.append([sess_avg_vis, sess_avg_bef, sess_avg_aft])

            pvals_groups.append([wilcoxon(vis, bef).pvalue,
                                 wilcoxon(vis, aft).pvalue,
                                 wilcoxon(bef, aft).pvalue])

    n_groups = len(group_labels)
    pvals_groups = np.array(pvals_groups)
    print(pvals_groups)

    phase_names = ["Visual delay", "Before peak", "After peak"]
    box_w       = 0.18
    group_gap   = 0.7
    offsets     = [-box_w, 0.0, box_w]
    group_centres = np.arange(n_groups) * group_gap

    # =====================================================================
    #  PLOT 1 — box-plots (main figure)
    # =====================================================================
    fig, ax = plt.subplots(figsize=(90*mm, 45*mm))

    for g_idx, (arrays, palette) in enumerate(zip(
            phase_arrays_all_groups, group_palettes)):
        pos = np.array(offsets) + group_centres[g_idx]

        bp = ax.boxplot(
            arrays,
            positions=pos,
            widths=box_w,
            patch_artist=True,
            whis=[0, 100],
            showfliers=False,
        )

        for phase_i in range(3):
            bp["boxes"][phase_i].set_facecolor(palette[phase_i])
            bp["boxes"][phase_i].set_linewidth(0)
            bp["boxes"][phase_i].set_alpha(0.8)

        for line_key in ("whiskers", "caps"):
            for line in bp[line_key]:
                line.set_color("k")
                line.set_linewidth(0.25)
        for line in bp["medians"]:
            line.set_color("k")
            line.set_linewidth(0.5)

    if show_significance:
        shift = 0.25
        for g_idx in range(n_groups):
            arrays = phase_arrays_all_groups[g_idx]
            y0 = max(arr.max() for arr in arrays) + 2
            gc = group_centres[g_idx]
            xv, xb, xa = gc - shift, gc, gc + shift
            for (xl, xr, pv, dy) in [ (xv, xb, pvals_groups[g_idx,0],0),
                                      (xv, xa, pvals_groups[g_idx,1],3),
                                      (xb, xa, pvals_groups[g_idx,2],6) ]:
                ax.plot([xl, xr], [y0+dy]*2, lw=0.7, color='k')
                ax.text((xl+xr)/2, y0+dy+0.4, _star(pv), ha='center', va='bottom', fontsize=5)

    ax.set_xticks(group_centres)
    ax.set_xticklabels(group_labels, fontsize=5, rotation=30, ha="right")
    ax.set_ylabel("% Neurons with discrepancy across conditions", fontsize=5)
    ax.set_ylim(-1, 30 if show_significance else 25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5, linewidth=0.25)
    plt.tight_layout()

    if save_fig:
        out = os.path.join(
            vis_dir, f"all_sessions_percent_neuron_discrepancy_box_{data_format}.pdf"
        )
        fig.savefig(out, bbox_inches=None, format='pdf', dpi=600)
        plt.close(fig)
    else:
        plt.show()

    # =====================================================================
    #  PLOT 2 — swarm plot of session averages (supplement)
    # =====================================================================
    rows = []
    for g_idx, (grp_lbl, palette, sess_avgs) in enumerate(zip(
            group_labels, group_palettes, session_avgs_all_groups)):
        for phase_i, phase_name in enumerate(phase_names):
            for val in sess_avgs[phase_i]:
                rows.append((grp_lbl, phase_name, float(val)))

    df_sess = pd.DataFrame(rows, columns=["Group", "Phase", "Percent"])

    # Build per-group palette dicts for seaborn (hue = Phase)
    palette_map = {}
    for phase_i, phase_name in enumerate(phase_names):
        # Use the first group's palette — all groups share the same index mapping
        # but each group has its own colours. seaborn swarmplot with hue doesn't
        # support per-x palette, so we draw group by group manually.
        pass

    fig2, ax2 = plt.subplots(figsize=(88*mm, 40*mm))

    dodge_shift = box_w  # same shift as box-plots for visual consistency

    for g_idx, (grp_lbl, palette, sess_avgs) in enumerate(zip(
            group_labels, group_palettes, session_avgs_all_groups)):
        gc = group_centres[g_idx]
        for phase_i, phase_name in enumerate(phase_names):
            x_center = gc + offsets[phase_i]
            vals = sess_avgs[phase_i]
            # small horizontal jitter for overlapping points
            jitter = np.random.default_rng(42).uniform(-box_w * 0.25, box_w * 0.25, size=len(vals))
            ax2.scatter(
                x_center + jitter,
                vals,
                s=12,
                color=palette[phase_i],
                edgecolors="k",
                linewidths=0,
                alpha=0.8,
                zorder=3,
            )

    ax2.set_xticks(group_centres)
    ax2.set_xticklabels(group_labels, fontsize=5, rotation=30, ha="right")
    ax2.set_xlabel("")
    ax2.set_ylabel("% Neurons with cross-condition discrepancy", fontsize=5, y=0.4)
    ax2.set_ylim(-1, 25)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", linestyle="--", alpha=0.5, linewidth=0.25)
    plt.tight_layout()

    if save_fig:
        out2 = os.path.join(
            vis_dir, f"all_sessions_percent_neuron_discrepancy_swarm_session_avg_{data_format}.pdf"
        )
        fig2.savefig(out2, bbox_inches=None, format='pdf', dpi=600)
        plt.close(fig2)
    else:
        plt.show()


def plot_number_dimension_discrepancy_all_sessions(
    session_info_dict: dict[str, dict],
    time_step: float,
    data_format: str,
    n_continuous_states,
    n_discrete_states,
    n_iters,
    model_type,
    dynamics_class,
    emission_class,
    init_type,
    alpha,
    *,
    significance_alpha: float = 0.05,
    correction_method: str = "fdr_bh",
    save_fig: bool = True,
):
    """Scatter-dot plots (jittered) with **custom palette per group** (no legend).

    Left two groups → 3 shades of *green*;  right two groups → 3 shades of *blue*.
    (Adjust the index logic below if group order changes.)
    """

    # ------------------------------------------------------------------
    group_labels, group_palettes = [], []
    phase_arrays_all_groups = []     # [[before, after], …]

    # ---- iterate groups ----------------------------------------------
    for info in session_info_dict.values():
        subject              = info["subject"]
        unit_filters         = info["unit_filters"]
        unit_filters_short   = info["unit_filters_short"]
        window_configs       = info["window_configs"]
        task                 = info["task"]
        task_short           = info["task_short"]
        trial_filters        = info["trial_filters"]
        visual_delay_times   = info["visual_delay_times"]
        peak_times           = info["peak_times"]

        for u_idx, (array, window_config,
                    visual_delay_time, peak_time) in enumerate(
                        zip(unit_filters, window_configs,
                            visual_delay_times, peak_times)):

            # ---------- load counts ------------------------------------
            data_name = '_'.join([
                'dynamical_latent_trajectories_per_dimension',
                subject,
                task,
                array,
                window_config,
                str(trial_filters),
                str(n_continuous_states),
                str(n_discrete_states),
                str(n_iters),
                model_type,
                dynamics_class,
                emission_class,
                init_type,
                str(alpha),
                'vdt' + str(visual_delay_time),
                'pt' + str(peak_time),
                'sig_counts'])

            res = np.load(os.path.join(results_dir, data_name + '.npz'))

            sig_counts_before_peak = res['sig_counts_before_peak']
            sig_counts_after_peak  = res['sig_counts_after_peak']

            sig_counts_before_peak = (
                np.sum(sig_counts_before_peak, axis=(1, 2))
                / n_continuous_states / 8 * 100.0
            )
            sig_counts_after_peak = (
                np.sum(sig_counts_after_peak, axis=(1, 2))
                / n_continuous_states / 8 * 100.0
            )

            # ---------- labels & palettes ------------------------------
            grp_lbl = f"{subject} {unit_filters_short[u_idx]} {task_short}"
            group_labels.append(grp_lbl)

            palette_3 = color_palettes[task][trial_filters[0]]  # 3 shades
            group_palettes.append([palette_3[0], palette_3[2]])                # keep 2

            phase_arrays_all_groups.append(
                [sig_counts_before_peak.flatten(),
                 sig_counts_after_peak.flatten()]
            )

    # ------------------------------------------------------------------
    # plotting
    fig, ax = plt.subplots(figsize=(90 * mm, 45 * mm))

    n_groups  = len(group_labels)
    phase_off = [-0.09, 0.09]   # x-offsets for before/after peak
    box_w     = 0.18            # for jitter scale

    rng = np.random.default_rng()

    for g_idx, (arrays, palette) in enumerate(
            zip(phase_arrays_all_groups, group_palettes)):

        for phase_i, phase_vals in enumerate(arrays):
            # jitter each point horizontally
            jitter = rng.uniform(-box_w / 3, box_w / 3, size=phase_vals.size)
            x_vals = (g_idx + phase_off[phase_i]) + jitter

            ax.scatter(
                x_vals,
                phase_vals,
                s=12,
                color=palette[phase_i],
                alpha=0.8,
                edgecolors='none',
                zorder=3,
            )

    # ------------------------------------------------------------------
    # cosmetics
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(group_labels, fontsize=5,
                       rotation=30, ha="right")
    ax.set_ylabel("% Dimensions with cross-condition discrepancy",
                  fontsize=5)
    ax.set_ylim(-1, 20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5, linewidth=0.25)

    plt.tight_layout()

    # save or show
    if save_fig:
        out = os.path.join(
            vis_dir,
            f"all_sessions_number_dimension_discrepancy_dots_{data_format}.pdf",
        )
        plt.savefig(out, bbox_inches=None, format="pdf", dpi=600)
        plt.close()
    else:
        plt.show()


# 8‑target ring layout → (row, col) in a 3×3 grid (centre omitted)
_IDX2POS = {0: (1, 2), 1: (0, 2), 2: (0, 1), 3: (0, 0),
            4: (1, 0), 5: (2, 0), 6: (2, 1), 7: (2, 2)}


# -----------------------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------------------

def plot_dynamical_latent_trajectories_per_dimension_ring(
    session_data_names,
    unit_filter,
    input_unit_filter,
    window_config,
    time_step,
    data_format,
    trial_filters,
    train_test_option,
    random_state,
    n_continuous_states,
    n_discrete_states,
    n_iters,
    model_type,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    visual_delay_time,
    peak_time,
    *,
    dims_to_plot=None,                 # <‑‑ NEW: list[int] or None (→ all)
    xtick_step: float = 0.4,          # <‑‑ NEW user‑tunable tick spacing (s)
    significance_alpha: float = 0.05,
    correction_method: str = "fdr_bh",
):
    """Plot 8‑target *ring* figures for selected latent dimensions.

    One PDF per requested latent dimension.  Each small panel shows mean ± SEM
    trajectories for *fast* (dashed) vs. *slow* (solid) conditions, colour‑
    coded by target.  A horizontal bar marks time spans with significant
    fast≠slow differences (after multiple‑comparison correction).

    Removed compared to the original routine
    ----------------------------------------
    1. Aggregate bar‑plot of significance counts.
    2. Optional NPZ save of significance arrays.
    3. Implicit loop over **all** dimensions – now selectable via *dims_to_plot*.
    """

    # 0) -------- input sanitisation ----------------------------------
    if dims_to_plot is None:
        dims_to_plot = list(range(n_continuous_states))
    else:
        dims_to_plot = sorted(set(dims_to_plot))
        bad = [d for d in dims_to_plot if d < 0 or d >= n_continuous_states]
        if bad:
            raise ValueError(f"dims_to_plot out of range: {bad}")

    fast_label, slow_label = trial_filters
    n_targets = 8

    # ------------------------------------------------------------------
    # 1) COLLECT trial‑wise continuous‑state arrays from **all sessions**
    # ------------------------------------------------------------------
    data_by_trial_filter_all = {
        fast_label: {"cs_by_target": [[] for _ in range(n_targets)]},
        slow_label: {"cs_by_target": [[] for _ in range(n_targets)]},
    }

    for session in session_data_names:
        dl = utils_processing.DataLoaderDuo(
            data_dir,
            results_dir,
            session,
            unit_filter,
            input_unit_filter,
            window_config,
            trial_filters,
        )
        dl.load_firing_rate_data()
        dl.load_cursor_data()
        dl.remove_target_overlap(target_radius=session_target_radii[session])
        dl.reformat_firing_rate_data(data_format, trial_length_filter_percentile=90)

        fast_dir, slow_dir = dl.get_model_result_dirs(
            time_offset=time_offset,
            train_test=train_test_option,
            data_format=data_format,
            model_type=model_type,
            dynamics_class=dynamics_class,
            emission_class=emission_class,
            init_type=init_type,
            subspace_type=subspace_type,
            alpha=alpha,
            check_existence=True,
        )

        tag = (f"r{random_state}_s{n_continuous_states}_i{n_iters}" if model_type == "LDS"
               else f"r{random_state}_s{n_continuous_states}_d{n_discrete_states}_i{n_iters}")

        with open(os.path.join(slow_dir, f"{tag}.pkl"), "rb") as f:
            res_slow = pickle.load(f)
        with open(os.path.join(fast_dir, f"{tag}.pkl"), "rb") as f:
            res_fast = pickle.load(f)

        tgt_fast, tgt_slow = dl.get_target_ids()
        tgt_fast -= tgt_fast.min();  tgt_slow -= tgt_slow.min()
        if data_format == "resample_avg":
            tgt_fast = tgt_slow = np.arange(n_targets)

        def _add(cs_list, tgt_array, label):
            for t in range(n_targets):
                data_by_trial_filter_all[label]["cs_by_target"][t].extend(
                    [cs for cs, keep in zip(cs_list, tgt_array == t) if keep])

        _add(res_fast["test_continuous_states"], tgt_fast, fast_label)
        _add(res_slow["test_continuous_states"], tgt_slow, slow_label)

    # ------------------------------------------------------------------
    # 2) SHARED time‑axis & significance masks
    # ------------------------------------------------------------------
    T_min = min(cs.shape[0] for lbl in trial_filters for tgt in data_by_trial_filter_all[lbl]["cs_by_target"] for cs in tgt)
    time_axis = np.arange(T_min) * time_step   # seconds relative to t0

    # Prepare nice tick positions in *seconds*
    t_last  = visual_delay_time + (T_min - 1) * time_step
    first_nice = np.ceil(visual_delay_time / xtick_step) * xtick_step
    all_ticks  = np.concatenate([[visual_delay_time], np.arange(first_nice, t_last + 1e-9, xtick_step)])
    tick_times = np.unique(np.round(all_ticks, 10))
    tick_pos   = tick_times - visual_delay_time   # because axis starts at 0 = t0

    # ipdb.set_trace()  # DEBUG: check tick positions

    sig_masks = defaultdict(dict)   # key = (dim, tgt) → bool[T]

    for d in dims_to_plot:
        for t in range(n_targets):
            fast_trials = [c[:T_min, d] for c in data_by_trial_filter_all[fast_label]["cs_by_target"][t]]
            slow_trials = [c[:T_min, d] for c in data_by_trial_filter_all[slow_label]["cs_by_target"][t]]
            if not fast_trials or not slow_trials:
                continue
            A, B = np.stack(fast_trials), np.stack(slow_trials)
            pvals = np.array([_p_value(A[:, i], B[:, i], use_paired=False) for i in range(T_min)])
            reject = (multipletests(pvals, alpha=significance_alpha, method=correction_method)[0]
                      if correction_method else pvals < significance_alpha)
            sig_masks[(d, t)] = reject

    # ------------------------------------------------------------------
    # 3) PLOT one ring per dimension
    # ------------------------------------------------------------------
    for d in dims_to_plot:
        # mean / sem per target & condition (also gather global y‑lim data)
        mean_d, sem_d = {}, {}
        y_min, y_max = np.inf, -np.inf

        for label in trial_filters:
            for t in range(n_targets):
                trials = [c[:T_min, d] for c in data_by_trial_filter_all[label]["cs_by_target"][t]]
                if not trials:
                    continue
                cs = np.stack(trials)
                m, s = cs.mean(0), cs.std(0, ddof=1) / np.sqrt(cs.shape[0])
                mean_d[(t, label)], sem_d[(t, label)] = m, s
                y_min = min(y_min, (m - s).min());  y_max = max(y_max, (m + s).max())

        # add head‑room for significance bar (5 %)
        y_headroom_frac = 0.05
        y_upper = y_max + y_headroom_frac * (y_max - y_min)

        fig, ax_grid = plt.subplots(3, 3, figsize=(45 * mm, 45 * mm), sharex=True, sharey=True)
        ax_grid = ax_grid.reshape(3, 3)

        for tgt in range(n_targets):
            r, c = _IDX2POS[tgt]
            ax = ax_grid[r, c]
            col = target_color_palette_8[tgt]

            for label, ls in zip(trial_filters, ["--", "-"]):
                m, s = mean_d[(tgt, label)], sem_d[(tgt, label)]
                ax.plot(time_axis, m, color=col, linestyle=ls, linewidth=size_line_thin, alpha=alpha_line)
                ax.fill_between(time_axis, m - s, m + s, color=col, alpha=alpha_line_thin, linewidth=0)

            # significance bar
            mask = sig_masks.get((d, tgt), np.zeros(T_min, bool))
            for s, e in _find_contiguous_true(mask):
                base = y_max + 0.03 * (y_max - y_min)
                ax.hlines(base, time_axis[s], time_axis[e], color="k", linewidth=size_line_thin, alpha=alpha_line)

            # décor
            ax.axvline(peak_time - visual_delay_time, color="black", linestyle=":", linewidth=size_line_thin, alpha=alpha_line)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        # --------------------------------------------------------------
        # GLOBAL, SHARED Y‑LIM & ASPECT  (once, outside inner loops)
        # --------------------------------------------------------------
        for ax in ax_grid.flatten():
            if ax.get_visible():
                ax.set_ylim(y_min, y_upper)
                ax.set_aspect((time_axis[-1] - time_axis[0]) / (y_upper - y_min))

        ax_grid[1, 1].set_visible(False)  # centre empty

        # tidy ticks
        label_ax = ax_grid[2, 0]
        for r in range(3):
            for c in range(3):
                ax = ax_grid[r, c]
                if ax is label_ax or not ax.get_visible():
                    continue
                ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False, bottom=False, left=False)

        label_ax.set_xlim(time_axis[0], time_axis[-1])
        # label_ax.set_ylim(y_min, y_max * 1.1)
        label_ax.set_xticks(tick_pos)
        label_ax.set_xticklabels([f"{tick_times[0]:.2f}"] + [f"{t:.1f}" for t in tick_times[1:]])
        label_ax.set_xlabel("Time (s)", fontsize=5);  label_ax.set_ylabel("Latent value (a.u.)", fontsize=5)

        fig.tight_layout()
        ses0 = session_data_names[0]     # for directory / naming
        out_dir = os.path.join(vis_dir, ses0)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"latent_dim{d}_ring_{ses0}.pdf"), dpi=600, transparent=True)
        plt.close(fig)



if __name__ == '__main__':

    # for (
    #     unit_filter,
    #     data_format) in itertools.product(
    #         unit_filters,
    #         data_formats):

    #     plot_behavioral_distances_to_target(
    #         unit_filter,
    #         data_format)

    #     plot_behavioral_target_acquisition_times(
    #         unit_filter,
    #         data_format)
        
        # plot_time_crossnobis_RDM_matrix(
        #     unit_filter, 
        #     data_format)

    # for data_format in data_formats:
        # plot_percent_neuron_direction_tuning_multi_session_anova_across_targets(
        #     data_format,
        #     visual_delay_time=0.1)
        
        # plot_percent_neuron_direction_tuning_multi_session_diff_from_baseline(
        #     data_format,
        #     pre_start_time_buffer=0.2,
        #     post_reach_time_buffer=0.5,
        #     visual_delay_time=0.1)


    direction_tuned_neuron_indices = {
        ## Neurons tuned to the same target across conditions found using ANOVA
        # 'sub-N1_ses-20190412_CenterStart' : [6, 16, 19, 23, 24, 25, 27, 38, 42, 52, 54, 56, 58, 69, 70, 76, 78, 79, 82, 95, 96, 98, 99, 101, 112, 117, 119, 121, 122, 124, 132],
        # 'sub-N1_ses-20190517_CenterStart' : [15, 19, 26, 27, 56, 58, 74, 77, 78, 79, 80, 82, 96, 99, 104, 113, 115, 122, 123],
        # 'sub-N1_ses-20190528_CenterStart' : [10, 12, 26, 40, 42, 52, 53, 65, 68, 79, 82, 95, 115, 123, 134, 136, 140, 145],

        ## Neurons tuned to the same target across conditions found using baseline (median, 15 degree threshold)
        # 'sub-N1_ses-20190412_CenterStart' : [2, 6, 14, 15, 16, 19, 23, 24, 25, 27, 38, 50, 54, 56, 65, 76, 78, 79, 80, 82, 96, 97, 99, 103, 106, 117, 119, 122, 124, 132],
        # 'sub-N1_ses-20190517_CenterStart' : [7, 13, 15, 22, 26, 27, 36, 39, 51, 75, 76, 77, 78, 79, 80, 101, 107, 121, 122, 123],
        # 'sub-N1_ses-20190528_CenterStart' : [10, 11, 12, 26, 27, 38, 40, 42, 52, 53, 62, 65, 68, 89, 99, 101, 123, 130, 134, 136],

        ## Neurons tuned to the same target across conditions found using baseline (mean, 15 degree threshold)
        # 'sub-N1_ses-20190412_CenterStart' : [2, 6, 14, 15, 16, 19, 23, 24, 25, 27, 33, 38, 50, 52, 53, 54, 56, 65, 72, 76, 78, 79, 80, 82, 96, 97, 99, 101, 103, 106, 117, 119, 122, 124, 132],
        # 'sub-N1_ses-20190517_CenterStart' : [13, 15, 19, 26, 27, 36, 39, 51, 52, 56, 75, 76, 77, 78, 79, 80, 100, 101, 107, 113, 121, 122, 123],
        # 'sub-N1_ses-20190528_CenterStart' : [10, 11, 12, 18, 26, 27, 30, 38, 40, 42, 52, 53, 54, 62, 65, 68, 77, 79, 89, 99, 101, 123, 134, 136, 142],

        ## Neurons tuned to the same target across conditions found using baseline (mean, 45 degree threshold)
        # 'sub-N1_ses-20190412_CenterStart' : [1, 2, 6, 14, 15, 16, 19, 22, 23, 24, 25, 27, 30, 33, 38, 41, 50, 51, 52, 53, 54, 56, 60, 61, 65, 70, 72, 75, 76, 77, 78, 79, 80, 81, 82, 92, 95, 96, 97, 98, 99, 101, 103, 106, 112, 113, 116, 117, 119, 122, 124, 128, 132],
        # 'sub-N1_ses-20190517_CenterStart' : [8, 13, 15, 17, 19, 26, 27, 29, 35, 36, 39, 48, 51, 52, 55, 56, 58, 65, 73, 74, 75, 76, 77, 78, 79, 80, 82, 99, 100, 101, 104, 107, 113, 115, 121, 122, 123, 125, 126],
        # 'sub-N1_ses-20190528_CenterStart' : [10, 11, 12, 14, 18, 19, 24, 26, 27, 30, 37, 38, 40, 42, 52, 53, 54, 62, 65, 68, 77, 79, 80, 82, 85, 89, 99, 101, 109, 117, 123, 134, 135, 136, 142, 145],

        ## Neurons tuned to only one condition found using baseline (median)
        # 'sub-N1_ses-20190412_CenterStart' : [7, 10, 12, 13, 22, 31, 36, 37, 40, 42, 44, 49, 57, 59, 67, 69, 72, 74, 83, 89, 102, 104, 109, 111, 116, 121, 130, 133],
        # 'sub-N1_ses-20190517_CenterStart' : [0, 1, 4, 5, 6, 14, 18, 20, 21, 28, 30, 31, 32, 34, 37, 38, 42, 46, 49, 53, 56, 59, 61, 62, 65, 67, 68, 69, 70, 72, 81, 84, 86, 87, 91, 96, 98, 102, 103, 105, 106, 108, 109, 110, 111, 112, 113, 114, 117, 126],
        # 'sub-N1_ses-20190528_CenterStart' : [0, 3, 4, 6, 7, 8, 15, 16, 20, 21, 24, 25, 28, 30, 32, 34, 36, 37, 39, 41, 43, 44, 45, 48, 50, 51, 55, 56, 57, 58, 60, 61, 64, 66, 67, 69, 70, 72, 74, 76, 77, 78, 79, 81, 83, 84, 87, 92, 93, 95, 96, 97, 100, 102, 103, 104, 105, 106, 107, 108, 110, 111, 112, 114, 116, 118, 120, 121, 122, 124, 125, 126, 128, 129, 131, 132, 133, 137, 139, 140, 141, 143, 144],
        
        ## Neurons tuned to only one condition found using baseline (mean)
        # 'sub-N1_ses-20190412_CenterStart' : [4, 7, 10, 12, 13, 29, 31, 34, 36, 37, 39, 40, 42, 44, 58, 59, 62, 66, 67, 69, 71, 73, 74, 83, 91, 102, 104, 108, 109, 111, 118, 120, 121, 123, 127, 129, 130, 133],
        # 'sub-N1_ses-20190517_CenterStart' : [0, 1, 4, 5, 12, 14, 18, 21, 22, 24, 28, 30, 31, 32, 34, 37, 38, 42, 46, 49, 53, 54, 59, 61, 62, 67, 68, 69, 70, 72, 81, 84, 85, 86, 87, 94, 96, 98, 102, 103, 105, 106, 108, 109, 110, 111, 112, 114, 117, 124],
        # 'sub-N1_ses-20190528_CenterStart' : [0, 3, 4, 6, 7, 8, 13, 15, 16, 17, 20, 22, 25, 28, 29, 32, 34, 35, 36, 39, 41, 43, 44, 45, 48, 50, 51, 55, 56, 57, 58, 61, 64, 66, 67, 69, 70, 72, 73, 74, 76, 78, 81, 83, 84, 87, 92, 93, 95, 96, 97, 98, 100, 102, 103, 104, 105, 106, 107, 108, 110, 111, 112, 113, 114, 116, 118, 120, 121, 122, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 137, 139, 140, 143, 144],
        
        ## Example neurons to show in the plot
        'sub-N1_ses-20190412_tf_CenterStart' : [37],
        'sub-N1_ses-20190517_tf_CenterStart' : [39],
        'sub-N1_ses-20190528_tf_CenterStart' : [15],
    }

    # target_tuned = {
        # 'sub-N1_ses-20190412_CenterStart' : [7, 3, 3, 2, 5, 5, 0, 0, 5, 0, 1, 4, 7, 7, 7, 1, 3, 0, 7, 2, 7, 2, 3, 3, 0, 7, 4, 7, 0, 7, 7],
        # 'sub-N1_ses-20190517_CenterStart' : [0, 0, 2, 4, 6, 1, 4, 1, 2, 1, 1, 1, 3, 0, 3, 7, 6, 1, 1],
        # 'sub-N1_ses-20190528_CenterStart' : [3, 6, 1, 4, 7, 2, 4, 2, 6, 6, 1, 3, 4, 7, 5, 2, 0, 5],

        # 'sub-N1_ses-20190517_CenterStart' : [0, 0, 0],
    # }


    for (
        session_data_name,
        unit_filter,
        data_format) in itertools.product(
            session_data_names,
            unit_filters,
            data_formats):

        # plot_direction_tuned_neurons_single_session(
        #     session_data_name,
        #     unit_filter,
        #     data_format,
        #     direction_tuned_neuron_indices=direction_tuned_neuron_indices[session_data_name],
        #     truncate_percentile=90,
        #     # target_tuned[session_data_name],
        #     target_tuned=None,
        #     visual_delay_time=0.0,
        #     # plot_pca=True,
        #     # plot_dpca=True,
        #     max_neurons_plot=None,
        #     plot_significance=True,
        #     significance_alpha=0.05)
        
    
        plot_neurons_by_target_single_session(
            session_data_name,
            unit_filter,
            data_format,
            direction_tuned_neuron_indices[session_data_name],
            truncate_percentile=90,
            # neuron_colors=['#B32578'],
            neuron_colors=['#FF78C9'],
            # neuron_colors=['#A286D9'],
            visual_delay_time=0.139,
            peak_time=0.188,
            significance_alpha=0.05,
            correction_method='fdr_bh')
        

    # for (
    #     unit_filter,
    #     data_format) in itertools.product(
    #         unit_filters,
    #         data_formats):
        
    #     plot_percent_neuron_discrepancy(
    #         session_data_names,
    #         unit_filter,
    #         data_format,
    #         truncate_percentile=90,
    #         reaction_times=[0.05] * 3,  # 50 ms reaction time
    #         # visual_delay_times=[0.139] * 3,  # 139 ms visual delay
    #         # visual_delay_times=[0.290] * 6,
    #         visual_delay_times=[0.294] * 3,
    #         # peak_times=[0.188] * 3,  # 188 ms peak time
    #         peak_times=[0.352] * 3,
    #         # peak_times=[0.354] * 6,
    #         significance_alpha=0.05,
    #         correction_method="fdr_bh",
    #         save_fig=True)
        
    
    # for (
    #     session_data_name,
    #     unit_filter,
    #     data_format) in itertools.product(
    #         session_data_names,
    #         unit_filters,
    #         data_formats):

    #     plot_cursor_trajectories_simple(
    #         session_data_name,
    #         unit_filter,
    #         data_format,
    #         show_average_trajectories=False,
    #         show_example_trajectories=True,
    #         show_target_positions=True)

        # plot_time_crossnobis_RDM_matrix_single_session(
        #     session_data_name,
        #     unit_filter, 
        #     data_format,
        #     pre_start_time_buffer=0.2,
        #     post_reach_time_buffer=0.5,
        #     ci_min=False,
        #     ci_max=False,
        #     pval=False,
        #     time_marker=0.25)

        # plot_neural_speed_single_session(
        #     session_data_name, 
        #     unit_filter, 
        #     data_format,
        #     truncate_percentile=90,
        #     pre_start_time_buffer=0.2,
        #     post_reach_time_buffer=0.5,
        #     time_marker=0.290,
        #     save_stats=False)


    # for (
    #     session_data_name,
    #     unit_filter,
    #     input_unit_filter,
    #     data_format,
    #     trial_filter,
    #     random_state,
    #     n_continuous_states,
    #     n_discrete_states,
    #     n_iters,
    #     model_type,
    #     dynamics_class,
    #     emission_class,
    #     init_type,
    #     subspace_type,
    #     alpha) in itertools.product(
    #         session_data_names,
    #         unit_filters,
    #         input_unit_filters,
    #         data_formats,
    #         trial_filters,
    #         random_states,
    #         ns_states,
    #         ns_discrete_states,
    #         ns_iters,
    #         model_types,
    #         dynamics_classes,
    #         emission_classes,
    #         init_types,
    #         subspace_types,
    #         alphas):
        
    #     plot_cursor_trajectories(
    #         session_data_name,
    #         unit_filter,
    #         input_unit_filter,
    #         data_format,
    #         trial_filter,
    #         random_state,
    #         n_continuous_states,
    #         n_discrete_states,
    #         n_iters,
    #         model_type,
    #         dynamics_class,
    #         emission_class,
    #         init_type,
    #         subspace_type,
    #         alpha,
    #         show_target_positions=True,
    #         discrete_state_overlay=True)


    # for (
    #     session_data_name,
    #     unit_filter,
    #     input_unit_filter,
    #     data_format,
    #     random_state,
    #     n_continuous_states,
    #     n_discrete_states,
    #     n_iters,
    #     model_type,
    #     dynamics_class,
    #     emission_class,
    #     init_type,
    #     subspace_type,
    #     alpha) in itertools.product(
    #         session_data_names,
    #         unit_filters,
    #         input_unit_filters,
    #         data_formats,
    #         random_states,
    #         ns_states,
    #         ns_discrete_states,
    #         ns_iters,
    #         model_types,
    #         dynamics_classes,
    #         emission_classes,
    #         init_types,
    #         subspace_types,
    #         alphas):
        
        # plot_3D_dynamical_latent_trajectories_integrated(
        #     session_data_name,
        #     unit_filter,
        #     input_unit_filter,
        #     data_format,
        #     random_state,
        #     n_continuous_states,
        #     n_discrete_states, 
        #     n_iters,
        #     model_type,
        #     dynamics_class,
        #     emission_class,
        #     init_type,
        #     subspace_type,
        #     alpha,
        #     show_individual_trajectories=False,
        #     show_average_trajectories=True,
        #     color_by_time_gradient=False,
        #     color_by_discrete_state=False,
        #     time_index_marker=None, ## 10 for 0.15s visual delay, 18 for no visual delay
        #     show_turning_points=False,
        #     show_flow_field=False,
        #     normalize_flow_field=False,
        #     show_flow_field_boundary=False,
        #     show_custom_axes=True)
        
        # plot_dynamical_latent_trajectories_per_dimension(
        #     session_data_name,
        #     unit_filter,
        #     input_unit_filter,
        #     data_format,
        #     random_state,
        #     n_continuous_states,
        #     n_discrete_states, 
        #     n_iters,
        #     model_type,
        #     dynamics_class,
        #     emission_class,
        #     init_type,
        #     subspace_type,
        #     alpha,
        #     peak_time=0.2,
        #     time_index_marker=None)

        # plot_discrete_states_over_time_single_session(
        #     session_data_name,
        #     unit_filter,
        #     input_unit_filter,
        #     data_format,
        #     random_state,
        #     n_continuous_states,
        #     n_discrete_states,
        #     n_iters,
        #     model_type,
        #     dynamics_class,
        #     emission_class,
        #     init_type,
        #     alpha,
        #     t_start=0)      


def plot_skew_symmetric_ratio(
    session_info,
    input_unit_filter,
    data_format,
    train_test_option,
    random_state,
    n_continuous_states,
    n_discrete_states,
    n_iters,
    model_type,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    plot_eigenspectra=True,
    plot_skew_ratio=True,
    plot_angular_velocity=True,
    plot_rotational_fraction=True,
    combine_trial_filters=False):
    """Plot rotational dynamics analysis across all subjects and tasks.

    Each panel is saved as a separate figure. All tasks/subjects are overlaid.

    Panel (a): Eigenspectra of M = A - I in the complex plane.
    Panel (b): Skew-symmetric ratio ||M_skew|| / ||M_sym||.
    Panel (c): Dominant angular velocity max|Im(lambda)|.
    Panel (d): Rotational fraction ||M_skew||^2_F / ||M||^2_F.

    Parameters
    ----------
    session_info : dict
        Full session_info dict with keys like 'N1_CenterStart', each containing
        'subject', 'task', 'session_data_names', 'unit_filters', 'window_configs',
        'trial_filters'.
    plot_eigenspectra, plot_skew_ratio, plot_angular_velocity, plot_rotational_fraction : bool
        Toggle which panels to generate.
    combine_trial_filters : bool
        If True, combine results across trial filters (e.g. fast/slow) within
        each group. Useful for joint models where dynamics matrices are shared
        across conditions — avoids plotting redundant identical bars. Occupancy
        is averaged across conditions.
    """
    from skew_symmetric_ratio_SLDS import (
        compute_skew_symmetric_ratio, compute_occupancy_weighted_skew_ratio,
        classify_states_by_temporal_order, compute_eigenspectra,
        compute_dominant_angular_velocity, compute_rotational_fraction)

    if combine_trial_filters and train_test_option != 'joint':
        raise ValueError(
            "combine_trial_filters=True is only valid with train_test_option='joint'. "
            "Separate models have different dynamics matrices per condition.")

    subject_markers = {'N1': '^', 'N2': 'o'}

    K = n_discrete_states if model_type not in ['LDS'] else 1
    if K == 2:
        state_labels = ['transient', 'steady']
    else:
        state_labels = [f'k={k}' for k in range(K)]

    ## ── Collect metrics across all session groups ──
    ## Each group is a (session_key, unit_filter) combination.
    ## Structure: list of dicts, each with metadata + per-trial_filter metric arrays.
    all_groups = []

    for session_key, info in session_info.items():
        subject = info['subject']
        task = info['task']
        group_trial_filters = info['trial_filters']

        for unit_filter, window_config in zip(info['unit_filters'], info['window_configs']):

            group = {
                'subject': subject,
                'task': task,
                'unit_filter': unit_filter,
                'trial_filters': group_trial_filters,
                'per_state_ratios': {tf: [] for tf in group_trial_filters},
                'per_state_occupancy': {tf: [] for tf in group_trial_filters},
                'per_state_angular_vel': {tf: [] for tf in group_trial_filters},
                'per_state_rot_frac': {tf: [] for tf in group_trial_filters},
                'eigenvalues': {tf: [] for tf in group_trial_filters},
            }

            for session_data_name in info['session_data_names']:

                data_loader = utils_processing.DataLoaderDuo(
                    data_dir,
                    results_dir,
                    session_data_name,
                    unit_filter,
                    input_unit_filter,
                    window_config,
                    group_trial_filters)

                data_loader.load_firing_rate_data()
                data_loader.load_cursor_data()
                data_loader.remove_target_overlap(
                    target_radius=session_target_radii[session_data_name])

                result_dirs = data_loader.get_model_result_dirs(
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

                ## Build model save name (no fold prefix for joint)
                if model_type in ['LDS']:
                    model_save_name = '_'.join([
                        'r' + str(random_state),
                        's' + str(n_continuous_states),
                        'i' + str(n_iters)])
                else:
                    model_save_name = '_'.join([
                        'r' + str(random_state),
                        's' + str(n_continuous_states),
                        'd' + str(n_discrete_states),
                        'i' + str(n_iters)])

                ## Load results for each trial filter
                res_per_tf = {}
                try:
                    for i_tf, tf in enumerate(group_trial_filters):
                        res_path = os.path.join(result_dirs[i_tf], model_save_name) + '.pkl'
                        with open(res_path, 'rb') as f:
                            res_per_tf[tf] = pickle.load(f)
                except FileNotFoundError:
                    print(f'Model not found: {res_path}')
                    continue

                ## Compute metrics for each trial filter
                for tf, res in res_per_tf.items():
                    model_obj = res['model']
                    As = model_obj.dynamics.As  # (K, D, D)

                    if model_type in ['LDS']:
                        ratio, _, _ = compute_skew_symmetric_ratio(As[0])
                        ang_vel = compute_dominant_angular_velocity(As[0])
                        rot_frac = compute_rotational_fraction(As[0])
                        eigs = compute_eigenspectra(As[0])

                        group['per_state_ratios'][tf].append(np.array([ratio]))
                        group['per_state_occupancy'][tf].append(np.array([1.0]))
                        group['per_state_angular_vel'][tf].append(np.array([ang_vel]))
                        group['per_state_rot_frac'][tf].append(np.array([rot_frac]))
                        group['eigenvalues'][tf].append([(eigs, 0, 1.0)])

                    else:
                        discrete_states = res['train_discrete_states']
                        K_model = As.shape[0]
                        _, psr, pso = compute_occupancy_weighted_skew_ratio(As, discrete_states)

                        ps_ang_vel = np.array([
                            compute_dominant_angular_velocity(As[k]) for k in range(K_model)])
                        ps_rot_frac = np.array([
                            compute_rotational_fraction(As[k]) for k in range(K_model)])

                        ## Reorder states by temporal center of mass
                        sorted_idx, _ = classify_states_by_temporal_order(
                            discrete_states, K_model)
                        psr = psr[sorted_idx]
                        pso = pso[sorted_idx]
                        ps_ang_vel = ps_ang_vel[sorted_idx]
                        ps_rot_frac = ps_rot_frac[sorted_idx]

                        group['per_state_ratios'][tf].append(psr)
                        group['per_state_occupancy'][tf].append(pso)
                        group['per_state_angular_vel'][tf].append(ps_ang_vel)
                        group['per_state_rot_frac'][tf].append(ps_rot_frac)

                        eig_entries = []
                        for sorted_k, orig_k in enumerate(sorted_idx):
                            eigs = compute_eigenspectra(As[orig_k])
                            eig_entries.append((eigs, sorted_k, pso[sorted_k]))
                        group['eigenvalues'][tf].append(eig_entries)

            all_groups.append(group)

    ## Check we have data
    has_data = any(
        len(g['per_state_ratios'][tf]) > 0
        for g in all_groups for tf in g['trial_filters'])
    if not has_data:
        print('No models found for any session. Skipping plot.')
        return

    ## ── Optionally combine trial filters ──
    ## For joint models the dynamics matrices are shared across conditions,
    ## so per-state metrics are identical.  Combining collapses redundant bars
    ## into a single set per group, averaging occupancy across conditions.
    if combine_trial_filters:
        combined_label = 'combined'
        for g in all_groups:
            tfs = g['trial_filters']
            metric_keys = ['per_state_ratios', 'per_state_angular_vel',
                           'per_state_rot_frac']

            ## Metrics: take from first trial filter (identical across conditions)
            first_tf = tfs[0]
            for mk in metric_keys:
                g[mk][combined_label] = g[mk][first_tf]

            ## Occupancy: average across conditions per session
            n_sessions = len(g['per_state_occupancy'][first_tf])
            combined_occ = []
            for i_sess in range(n_sessions):
                occ_stack = []
                for tf in tfs:
                    if i_sess < len(g['per_state_occupancy'][tf]):
                        occ_stack.append(g['per_state_occupancy'][tf][i_sess])
                combined_occ.append(np.mean(occ_stack, axis=0))
            g['per_state_occupancy'][combined_label] = combined_occ

            ## Eigenvalues: take from first trial filter
            g['eigenvalues'][combined_label] = g['eigenvalues'][first_tf]

            ## Replace trial_filters with combined
            g['trial_filters'] = [combined_label]

    ## ── Determine task ordering and gather unique (task, trial_filter) pairs ──
    task_order = ['CenterStart', 'CenterStartInterleave', 'RadialGrid']
    task_tf_pairs = []  # list of (task, trial_filter) in display order
    for task in task_order:
        task_groups = [g for g in all_groups if g['task'] == task]
        if not task_groups:
            continue
        for tf in task_groups[0]['trial_filters']:
            task_tf_pairs.append((task, tf))

    ## ── Build group labels (same format as plot_percent_neuron_discrepancy_all_sessions) ──
    subject_display = {'N1': 'JJ', 'N2': 'RD'}
    task_short_map = {'CenterStart': 'CO', 'CenterStartInterleave': 'COI', 'RadialGrid': 'RG'}

    ## Assign display labels to each group
    for g in all_groups:
        subj_disp = subject_display.get(g['subject'], g['subject'])
        uf_short = g['unit_filter'].replace('MC-LAT', 'MCL').replace('MC-MED', 'MCM')
        task_short = task_short_map.get(g['task'], g['task'])
        g['label'] = f"{subj_disp} {uf_short} {task_short}"

    ## ── Helper: dot + mean-line panel ──
    ## Groups are (subject, unit_filter, task), matching the x-axis style of
    ## plot_percent_neuron_discrepancy_all_sessions.  Within each group,
    ## columns for each (trial_filter × discrete_state) are side-by-side.
    ## No trial-filter / state labels — user will add in Illustrator.
    def _plot_bar_panel(ax, metric_key, ylabel, ylim=None, ylabel_fontsize=6):
        slot_width = 0.25
        group_gap = 0.8
        section_gap = 0.2  # extra spacing between subject-task sections

        ## Compute group centres with gaps between (subject, task) sections
        group_centres = []
        x_pos_running = 0.0
        prev_subject = None
        prev_task = None
        for g in all_groups:
            if prev_task is not None and (g['task'] != prev_task or g['subject'] != prev_subject):
                x_pos_running += section_gap
            group_centres.append(x_pos_running)
            x_pos_running += group_gap
            prev_task = g['task']
            prev_subject = g['subject']
        group_centres = np.array(group_centres)

        for g_idx, g in enumerate(all_groups):
            task = g['task']
            gc = group_centres[g_idx]
            n_slots = len(g['trial_filters']) * K

            ## Slot offsets centered around gc
            offsets = np.linspace(
                -(n_slots - 1) * slot_width / 2,
                 (n_slots - 1) * slot_width / 2,
                 n_slots)

            slot_i = 0
            for tf in g['trial_filters']:
                values_list = g[metric_key].get(tf, [])
                occ_list = g['per_state_occupancy'].get(tf, [])

                if not values_list:
                    slot_i += K
                    continue

                values = np.array(values_list)       # (n_sessions, K)
                occupancy = np.array(occ_list)
                n_sessions = values.shape[0]

                mean_vals = np.nanmean(values, axis=0)
                mean_occ = np.mean(occupancy, axis=0)

                ## For 'combined' trial filter, use first original trial filter's colors
                task_colors = discrete_state_colors[task]
                colors = task_colors.get(tf, list(task_colors.values())[0])

                for k in range(K):
                    x_center = gc + offsets[slot_i]
                    color = colors[k % len(colors)]
                    half_w = slot_width * 0.35

                    ## Mean line (slightly darker than dots)
                    ax.hlines(mean_vals[k],
                              x_center - half_w, x_center + half_w,
                              color=color, alpha=1, linewidth=1.2, zorder=4)

                    ## Individual data points
                    jitter = np.random.default_rng(42 + g_idx * 100 + slot_i).uniform(
                        -half_w * 0.5, half_w * 0.5, size=n_sessions)
                    ax.scatter(
                        x_center + jitter,
                        values[:, k],
                        s=20, color=color,
                        edgecolors='k', linewidths=0,
                        alpha=0.7, zorder=3)

                    ## Occupancy annotation — nudge left/right within pairs
                    if n_discrete_states > 1:
                        y_top = np.nanmax(values[:, k]) if n_sessions > 0 else mean_vals[k]
                        x_nudge = -slot_width * 0.05 if slot_i % 2 == 0 else slot_width * 0.05
                        ha = 'right' if slot_i % 2 == 0 else 'left'
                        ax.text(x_center + x_nudge,
                                y_top + 0.1 * max(abs(np.nanmax(mean_vals)), 1e-6),
                                f'{mean_occ[k]:.0%}',
                                ha=ha, va='bottom', fontsize=5)

                    slot_i += 1

        ax.set_xticks(group_centres)
        ax.set_xticklabels([g['label'] for g in all_groups],
                           fontsize=5, rotation=30, ha='right')
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if ylim is not None:
            ax.set_ylim(ylim)

    ## ── Build base filename ──
    base_name = '_'.join(map(str, [x for x in [
        'all_subjects',
        'skew_analysis',
        data_format,
        train_test_option,
        'combined' if combine_trial_filters else None,
        model_type,
        n_continuous_states,
        n_discrete_states,
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))

    ## ── Panel (a): Eigenspectra ──
    if plot_eigenspectra:
        fig_eig, ax_eig = plt.subplots(1, 1, figsize=(50*mm, 45*mm))
        ax_eig.axhline(y=0, color='gray', lw=0.3, ls='-')
        ax_eig.axvline(x=0, color='gray', lw=0.3, ls='--')

        legend_handles = []
        for group in all_groups:
            marker = subject_markers.get(group['subject'], 'o')
            for tf in group['trial_filters']:
                task_colors = discrete_state_colors[group['task']]
                colors = task_colors.get(tf, list(task_colors.values())[0])
                for session_eigs in group['eigenvalues'][tf]:
                    for eigs, state_type, occ in session_eigs:
                        color = colors[state_type % len(colors)]
                        marker_size = max(2, occ * 30)
                        ax_eig.scatter(
                            eigs.real, eigs.imag,
                            c=[color], s=marker_size, alpha=0.6,
                            marker=marker, edgecolors='none', zorder=2)

        ## Build legend: one entry per subject marker + one per task color
        for subj, mkr in subject_markers.items():
            subj_groups = [g for g in all_groups if g['subject'] == subj]
            if subj_groups:
                legend_handles.append(plt.Line2D(
                    [0], [0], marker=mkr, color='gray', linestyle='None',
                    markersize=4, label=subj))
        for task in task_order:
            task_groups = [g for g in all_groups if g['task'] == task]
            if task_groups:
                tf0 = task_groups[0]['trial_filters'][0]
                color = discrete_state_colors[task][tf0][0]
                task_short = {'CenterStart': 'CO', 'CenterStartInterleave': 'COI',
                              'RadialGrid': 'RG'}.get(task, task)
                legend_handles.append(plt.Line2D(
                    [0], [0], marker='s', color=color, linestyle='None',
                    markersize=4, label=task_short))

        ax_eig.legend(handles=legend_handles, fontsize=4, loc='upper left',
                      frameon=False, handletextpad=0.3, borderpad=0.2)
        ax_eig.set_xlabel('Re', fontsize=6)
        ax_eig.set_ylabel('Im', fontsize=6)
        ax_eig.set_aspect('equal')
        ax_eig.tick_params(axis='both', which='major', labelsize=5)
        ax_eig.spines['top'].set_visible(False)
        ax_eig.spines['right'].set_visible(False)

        fig_eig.tight_layout(pad=0.5)
        save_path = os.path.join(vis_dir, base_name + '_eigenspectra.pdf')
        fig_eig.savefig(save_path, dpi=600, transparent=True, bbox_inches='tight', format='pdf')
        plt.close(fig_eig)
        print(f'Saved eigenspectra plot to {save_path}')

    ## ── Panel (b): Skew-symmetric ratio ──
    if plot_skew_ratio:
        fig_ratio, ax_ratio = plt.subplots(1, 1, figsize=(90*mm, 45*mm))
        _plot_bar_panel(ax_ratio, 'per_state_ratios',
                        r'Skew-symmetric ratio ($\|M_{\mathrm{skew}}\| \,/\, \|M_{\mathrm{sym}}\|$)',
                        ylabel_fontsize=5)
        fig_ratio.tight_layout(pad=0.5)
        fig_ratio.subplots_adjust(bottom=0.25)
        save_path = os.path.join(vis_dir, base_name + '_skew_ratio.pdf')
        fig_ratio.savefig(save_path, dpi=600, transparent=True, bbox_inches='tight', format='pdf')
        plt.close(fig_ratio)
        print(f'Saved skew ratio plot to {save_path}')

    ## ── Panel (c): Dominant angular velocity (convert rad/bin → rad/s) ──
    if plot_angular_velocity:
        ## Convert rad/bin to rad/s
        for g in all_groups:
            g['per_state_angular_vel_per_s'] = {
                tf: [v / time_step for v in g['per_state_angular_vel'][tf]]
                for tf in g['trial_filters']}

        fig_angvel, ax_angvel = plt.subplots(1, 1, figsize=(90*mm, 45*mm))
        _plot_bar_panel(ax_angvel, 'per_state_angular_vel_per_s',
                        'Angular velocity (rad/s)')
        fig_angvel.tight_layout(pad=0.5)
        fig_angvel.subplots_adjust(bottom=0.25)
        save_path = os.path.join(vis_dir, base_name + '_angular_velocity.pdf')
        fig_angvel.savefig(save_path, dpi=600, transparent=True, bbox_inches='tight', format='pdf')
        plt.close(fig_angvel)
        print(f'Saved angular velocity plot to {save_path}')

    ## ── Panel (d): Rotational fraction ──
    if plot_rotational_fraction:
        fig_rotfrac, ax_rotfrac = plt.subplots(1, 1, figsize=(90*mm, 45*mm))
        _plot_bar_panel(ax_rotfrac, 'per_state_rot_frac',
                        r'$\|M_{\mathrm{skew}}\|^2_F \,/\, \|M\|^2_F$',
                        ylim=(0, 1))
        fig_rotfrac.tight_layout(pad=0.5)
        fig_rotfrac.subplots_adjust(bottom=0.25)
        save_path = os.path.join(vis_dir, base_name + '_rotational_fraction.pdf')
        fig_rotfrac.savefig(save_path, dpi=600, transparent=True, bbox_inches='tight', format='pdf')
        plt.close(fig_rotfrac)
        print(f'Saved rotational fraction plot to {save_path}')


    # for (
    #     unit_filter,
    #     input_unit_filter,
    #     data_format,
    #     random_state,
    #     n_continuous_states,
    #     n_discrete_states,
    #     n_iters,
    #     model_type,
    #     dynamics_class,
    #     emission_class,
    #     init_type,
    #     subspace_type,
    #     alpha) in itertools.product(
    #         unit_filters,
    #         input_unit_filters,
    #         data_formats,
    #         random_states,
    #         ns_states,
    #         ns_discrete_states,
    #         ns_iters,
    #         model_types,
    #         dynamics_classes,
    #         emission_classes,
    #         init_types,
    #         subspace_types,
    #         alphas):

    #     plot_discrete_states_over_time(
    #         unit_filter,
    #         input_unit_filter,
    #         data_format,
    #         random_state,
    #         n_continuous_states,
    #         n_discrete_states,
    #         n_iters,
    #         model_type,
    #         dynamics_class,
    #         emission_class,
    #         init_type,
    #         alpha)
        