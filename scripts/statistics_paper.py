import os

import pickle
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import (t, wilcoxon, mannwhitneyu, ttest_1samp, ttest_rel,
                         f_oneway, ttest_ind)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from statsmodels.stats.multitest import multipletests

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cross_decomposition import PLSRegression

import utils.utils_processing as utils_processing
import utils.utils_decoding as utils_decoding
import utils.utils_vis as utils_vis
import scripts.config as config
from visualizations.vis_config import *



## Read parameters from config
data_dir           = config.data_dir
results_dir        = config.results_dir
vis_dir            = config.vis_dir
session_data_names = config.session_data_names

unit_filters       = config.unit_filters
input_unit_filters = config.input_unit_filters
window_config      = config.window_config
time_offset        = config.time_offset
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


def _per_target_crossnobis(fr, target_angles, time_step):
    """Compute crossnobis per target, return stacked (n_targets, T, T)."""
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
    tgt_mats = [m[:int(tgt_min_T), :int(tgt_min_T)] for m in tgt_mats]
    return np.stack(tgt_mats)  # (n_targets, T, T)


def neural_speed_statistics(
    session_data_names: str,
    unit_filters: str,
    window_config: str,
    time_step: float,
    trial_filters: list[str],
    truncate_percentile: int = 90,
    pre_start_time_buffer: float = 0.2,
    post_reach_time_buffer: float = 0.5,
    superdiagonal_order: int = 1,
    per_target: bool = False):
    

    def analyze_neural_speed(
        neural_speed_truncate_end: np.ndarray,
        neural_speed_truncate_front: np.ndarray):

        # neural_speed has shape (n_trials, n_time_bins)
        baseline_window = int(0.2 / time_step)
        baseline_trial  = neural_speed_truncate_end[:, :baseline_window].mean(axis=1)

        peak_trial      = neural_speed_truncate_end.max(axis=1)
        return_trial    = neural_speed_truncate_front[:, -1]

        # ─── Descriptive means ────────────────────────────────────────────────
        print(f'Baseline mean: {baseline_trial.mean():.3f}, '
              f'Peak mean: {peak_trial.mean():.3f}, '
              f'Return mean: {return_trial.mean():.3f}')

        # ─── Significance tests ──────────────────────────────────────────────
        w_peak,  p_peak  = wilcoxon(peak_trial,   baseline_trial, alternative='greater')
        w_ret,   p_ret   = wilcoxon(return_trial, baseline_trial, alternative='two-sided')
        t_peak,  p_tpeak = ttest_rel(peak_trial,  baseline_trial)
        t_ret,   p_tret  = ttest_rel(return_trial, baseline_trial)

        print(f'Peak vs Baseline:  W={w_peak:.0f}, p={p_peak:.3g}  (t={t_peak:.3f}, p={p_tpeak:.3g})')
        print(f'Return vs Baseline: W={w_ret:.0f}, p={p_ret:.3g}  (t={t_ret:.3f}, p={p_tret:.3g})')

        # ─── Effect size + 95 % CI for Peak vs Baseline ──────────────────────
        diff_peak = peak_trial - baseline_trial
        n         = diff_peak.size
        mean_diff = diff_peak.mean()
        sd_diff   = diff_peak.std(ddof=1)
        d_peak    = mean_diff / sd_diff                      # Cohen's d_z

        t_crit    = t.ppf(0.975, df=n-1)                     # two-sided 95 % CI
        half_w    = t_crit * sd_diff / np.sqrt(n)
        ci_peak   = (mean_diff - half_w, mean_diff + half_w)

        print(f'Peak vs Baseline:  Mean Δ={mean_diff:.3f} '
              f'(95 % CI [{ci_peak[0]:.3f}, {ci_peak[1]:.3f}]), '
              f"Cohen's d = {d_peak:.3f}")

        # ─── Effect size + 95 % CI for Return vs Baseline ────────────────────
        diff_ret  = return_trial - baseline_trial
        n         = diff_ret.size
        mean_diff = diff_ret.mean()
        sd_diff   = diff_ret.std(ddof=1)
        d_ret     = mean_diff / sd_diff                      # Cohen's d_z

        t_crit    = t.ppf(0.975, df=n-1)
        half_w    = t_crit * sd_diff / np.sqrt(n)
        ci_ret    = (mean_diff - half_w, mean_diff + half_w)

        print(f'Return vs Baseline: Mean Δ={mean_diff:.3f} '
              f'(95 % CI [{ci_ret[0]:.3f}, {ci_ret[1]:.3f}]), '
              f"Cohen's d = {d_ret:.3f}")


    def analyze_neural_speed_all_conditions(
        neural_speed_all_truncate_end,
        neural_speed_all_truncate_front):

        assert len(neural_speed_all_truncate_end) == len(neural_speed_all_truncate_front) == len(trial_filters)

        baseline_trials_all = []
        peak_trials_all     = []
        return_trials_all   = []

        for i_tf, trial_filter in enumerate(trial_filters):
            neural_speed_truncate_end = neural_speed_all_truncate_end[i_tf]
            neural_speed_truncate_front = neural_speed_all_truncate_front[i_tf]

            print(f'Analyzing {trial_filter} trials:')

            # neural_speed has shape (n_trials, n_time_bins)
            baseline_window = int(0.2 / time_step)
            baseline_trial  = neural_speed_truncate_end[:, :baseline_window].mean(axis=1)

            peak_trial      = neural_speed_truncate_end.max(axis=1)
            return_trial    = neural_speed_truncate_front[:, -1]

            baseline_trials_all.append(baseline_trial)
            peak_trials_all.append(peak_trial)
            return_trials_all.append(return_trial)

        baseline_trial = np.concatenate(baseline_trials_all)
        peak_trial     = np.concatenate(peak_trials_all)
        return_trial   = np.concatenate(return_trials_all)

        # ─── Descriptive means ────────────────────────────────────────────────
        print(f'Baseline mean: {baseline_trial.mean():.3f}, '
              f'Peak mean: {peak_trial.mean():.3f}, '
              f'Return mean: {return_trial.mean():.3f}')

        # ─── Significance tests ──────────────────────────────────────────────
        w_peak,  p_peak  = wilcoxon(peak_trial,   baseline_trial, alternative='greater')
        w_ret,   p_ret   = wilcoxon(return_trial, baseline_trial, alternative='two-sided')
        t_peak,  p_tpeak = ttest_rel(peak_trial,  baseline_trial)
        t_ret,   p_tret  = ttest_rel(return_trial, baseline_trial)

        print(f'Peak vs Baseline:  W={w_peak:.0f}, p={p_peak:.3g}  (t={t_peak:.3f}, p={p_tpeak:.3g})')
        print(f'Return vs Baseline: W={w_ret:.0f}, p={p_ret:.3g}  (t={t_ret:.3f}, p={p_tret:.3g})')

        # ─── Effect size + 95 % CI for Peak vs Baseline ──────────────────────
        diff_peak = peak_trial - baseline_trial
        n         = diff_peak.size
        mean_diff = diff_peak.mean()
        sd_diff   = diff_peak.std(ddof=1)
        d_peak    = mean_diff / sd_diff                      # Cohen's d_z

        t_crit    = t.ppf(0.975, df=n-1)                     # two-sided 95 % CI
        half_w    = t_crit * sd_diff / np.sqrt(n)
        ci_peak   = (mean_diff - half_w, mean_diff + half_w)

        print(f'Peak vs Baseline:  Mean Δ={mean_diff:.3f} '
              f'(95 % CI [{ci_peak[0]:.3f}, {ci_peak[1]:.3f}]), '
              f"Cohen's d = {d_peak:.3f}")

        # ─── Effect size + 95 % CI for Return vs Baseline ────────────────────
        diff_ret  = return_trial - baseline_trial
        n         = diff_ret.size
        mean_diff = diff_ret.mean()
        sd_diff   = diff_ret.std(ddof=1)
        d_ret     = mean_diff / sd_diff                      # Cohen's d_z

        t_crit    = t.ppf(0.975, df=n-1)
        half_w    = t_crit * sd_diff / np.sqrt(n)
        ci_ret    = (mean_diff - half_w, mean_diff + half_w)

        print(f'Return vs Baseline: Mean Δ={mean_diff:.3f} '
              f'(95 % CI [{ci_ret[0]:.3f}, {ci_ret[1]:.3f}]), '
              f"Cohen's d = {d_ret:.3f}")


    def analyze_neural_speed_phase(
        neural_speeds,
        full_trial_lengths):
        
        neural_speed_fast = neural_speeds[0]
        neural_speed_slow = neural_speeds[1]
        full_trial_lengths_fast = full_trial_lengths[0]
        full_trial_lengths_slow = full_trial_lengths[1]

        # Compute the steady phase durations
        steady_phase_durations_fast = (full_trial_lengths_fast - np.argmax(neural_speed_fast, axis=1)) * time_step
        steady_phase_durations_slow = (full_trial_lengths_slow - np.argmax(neural_speed_slow, axis=1)) * time_step

        # 95 % parametric CI for the mean
        from scipy.stats import t, mannwhitneyu, ttest_ind
        def mean_ci(vec):
            n  = vec.size
            sd = vec.std(ddof=1)
            halfw = t.ppf(0.975, df=n-1) * sd / np.sqrt(n)
            return vec.mean()-halfw, vec.mean()+halfw

        ci_fast = mean_ci(steady_phase_durations_fast)
        ci_slow = mean_ci(steady_phase_durations_slow)

        print(f"Ballistic steady duration: {np.mean(steady_phase_durations_fast):.3f}s  (95 % CI {ci_fast[0]:.3f}-{ci_fast[1]:.3f})")
        print(f"Sustained steady duration: {np.mean(steady_phase_durations_slow):.3f}s  (95 % CI {ci_slow[0]:.3f}-{ci_slow[1]:.3f})")

        # ─── significance test ────────────────────────────────────────────
        u, p_u   = mannwhitneyu(steady_phase_durations_fast, steady_phase_durations_slow, alternative='two-sided')
        tval, p_t = ttest_ind(steady_phase_durations_fast, steady_phase_durations_slow, equal_var=False)        # Welch

        print(f"Fast vs slow: U={u:.1f}, p={p_u:.3g}   (Welch t={tval:.2f}, p={p_t:.3g})")

        # ─── effect size: rank-biserial r for MWU ─────────────────────────
        n1, n2 = len(steady_phase_durations_fast), len(steady_phase_durations_slow)
        r_rb = 1 - 2 * u / (n1 * n2)
        print(f"Rank-biserial r = {r_rb:.3f}")

        # ─── 95 % CI for mean difference (fast - slow) ──────────────────
        mean_diff = np.mean(steady_phase_durations_fast) - np.mean(steady_phase_durations_slow)
        se_diff = np.sqrt(steady_phase_durations_fast.var(ddof=1) / n1
                          + steady_phase_durations_slow.var(ddof=1) / n2)
        df_eff = (steady_phase_durations_fast.var(ddof=1) / n1
                  + steady_phase_durations_slow.var(ddof=1) / n2)**2 / (
            (steady_phase_durations_fast.var(ddof=1) / n1)**2 / (n1 - 1)
            + (steady_phase_durations_slow.var(ddof=1) / n2)**2 / (n2 - 1))
        margin = t.ppf(0.975, df_eff) * se_diff
        print(f"Mean Δ (fast - slow) = {mean_diff:.3f} s "
              f"(95 % CI [{mean_diff - margin:.3f}, {mean_diff + margin:.3f}])")


    pre_start_idx = int(pre_start_time_buffer / time_step)
    post_reach_idx = int(post_reach_time_buffer / time_step)

    crossnobis_matrices_all_truncate_end   = [[] for _ in trial_filters]
    crossnobis_matrices_all_truncate_front = [[] for _ in trial_filters]
    full_trial_lengths_all               = [[] for _ in trial_filters]
    min_lengths_truncate_end             = [np.inf] * len(trial_filters)
    min_lengths_truncate_front           = [np.inf] * len(trial_filters)

    # ─────────────── gather matrices over sessions ───────────────
    for session_data_name in session_data_names:
        for unit_filter in unit_filters:
            for i_tf, trial_filter in enumerate(trial_filters):

                # Load results if they exist.
                # '_stats_per_target' avoids colliding with vis_paper's
                # '_per_target' cache (different keys).
                pt_suffix = '_stats_per_target' if per_target else ''
                crossnobis_save_path = os.path.join(
                    results_dir,
                    session_data_name,
                    'crossnobis_matrices',
                    unit_filter,
                    window_config,
                    trial_filter + pt_suffix + '.pkl'
                )
                print(f'Processing {session_data_name}, {unit_filter}, {trial_filter}')
                expected_keys = {'crossnobis_matrices_truncate_end',
                                 'crossnobis_matrices_truncate_front',
                                 'full_trial_lengths'}
                if os.path.exists(crossnobis_save_path):
                    with open(crossnobis_save_path, 'rb') as f:
                        data = pickle.load(f)
                    if expected_keys <= data.keys():
                        crossnobis_matrices_truncate_end = data['crossnobis_matrices_truncate_end']
                        crossnobis_matrices_truncate_front = data['crossnobis_matrices_truncate_front']
                        full_trial_lengths = data['full_trial_lengths']

                        crossnobis_matrices_all_truncate_end[i_tf].append(crossnobis_matrices_truncate_end)
                        crossnobis_matrices_all_truncate_front[i_tf].append(crossnobis_matrices_truncate_front)
                        full_trial_lengths_all[i_tf].append(full_trial_lengths)

                        min_lengths_truncate_end[i_tf] = min(min_lengths_truncate_end[i_tf], crossnobis_matrices_truncate_end.shape[1])
                        min_lengths_truncate_front[i_tf] = min(min_lengths_truncate_front[i_tf], crossnobis_matrices_truncate_front.shape[1])

                        continue
                    else:
                        print(f'    Cache missing expected keys, recomputing...')

                # Load data
                dl = utils_processing.DataLoader(
                    data_dir,
                    results_dir,
                    session_data_name,
                    unit_filter,
                    None,
                    window_config,
                    trial_filter)
                dl.load_firing_rate_data()

                fr_simple_truncate_end, *_ = dl.reformat_firing_rate_data(
                    data_format='truncate_end',
                    index_buffer=post_reach_idx,
                    trial_length_filter_percentile=truncate_percentile,
                )

                fr_simple_truncate_end = fr_simple_truncate_end[:, :-post_reach_idx, :]

                # Get target angles before second reformat modifies dl state
                if per_target:
                    target_angles_end = dl.get_target_angles()

                _, _, _, full_trial_lengths, _, _ = dl.reformat_firing_rate_data(
                    data_format=None,
                    index_buffer=post_reach_idx,
                )
                full_trial_lengths -= post_reach_idx # remove the post-reach buffer

                dl = utils_processing.DataLoader(
                    data_dir,
                    results_dir,
                    session_data_name,
                    unit_filter,
                    None,
                    window_config,
                    trial_filter)
                dl.load_firing_rate_data()

                fr_simple_truncate_front, *_ = dl.reformat_firing_rate_data(
                    data_format='truncate_front',
                    index_buffer=post_reach_idx,
                    trial_length_filter_percentile=truncate_percentile,
                )

                fr_simple_truncate_front = fr_simple_truncate_front[:, pre_start_idx:-post_reach_idx, :]

                if per_target:
                    target_angles_front = dl.get_target_angles()

                # Crossnobis -------------------------------------------------------
                if per_target:
                    crossnobis_matrices_truncate_end = _per_target_crossnobis(
                        fr_simple_truncate_end, target_angles_end, time_step)
                    crossnobis_matrices_truncate_front = _per_target_crossnobis(
                        fr_simple_truncate_front, target_angles_front, time_step)
                else:
                    _, crossnobis_matrices_truncate_end = utils_processing.compute_crossnobis_matrix(
                        fr_simple_truncate_end, time_step=time_step)
                    _, crossnobis_matrices_truncate_front = utils_processing.compute_crossnobis_matrix(
                        fr_simple_truncate_front, time_step=time_step)

                crossnobis_matrices_all_truncate_end[i_tf].append(crossnobis_matrices_truncate_end)
                crossnobis_matrices_all_truncate_front[i_tf].append(crossnobis_matrices_truncate_front)
                full_trial_lengths_all[i_tf].append(full_trial_lengths)

                min_lengths_truncate_end[i_tf] = min(min_lengths_truncate_end[i_tf], crossnobis_matrices_truncate_end.shape[1])
                min_lengths_truncate_front[i_tf] = min(min_lengths_truncate_front[i_tf], crossnobis_matrices_truncate_front.shape[1])

                # Save the crossnobis matrices
                
                os.makedirs(os.path.dirname(crossnobis_save_path), exist_ok=True)
                with open(crossnobis_save_path, 'wb') as f:
                    pickle.dump({
                        'crossnobis_matrices_truncate_end': crossnobis_matrices_truncate_end,
                        'crossnobis_matrices_truncate_front': crossnobis_matrices_truncate_front,
                        'full_trial_lengths': full_trial_lengths
                    }, f)


    # ─────────────── truncate to common size & average ───────────────
    for i_tf in range(len(trial_filters)):
        for i in range(len(crossnobis_matrices_all_truncate_end[i_tf])):
            session_crossnobis_matrices = crossnobis_matrices_all_truncate_end[i_tf][i]
            crossnobis_matrices_all_truncate_end[i_tf][i] = session_crossnobis_matrices[:, :min_lengths_truncate_end[i_tf], :min_lengths_truncate_end[i_tf]]
        for i in range(len(crossnobis_matrices_all_truncate_front[i_tf])):
            session_crossnobis_matrices = crossnobis_matrices_all_truncate_front[i_tf][i]
            crossnobis_matrices_all_truncate_front[i_tf][i] = session_crossnobis_matrices[:, -min_lengths_truncate_front[i_tf]:, -min_lengths_truncate_front[i_tf]:]

    crossnobis_matrices_all_truncate_end   = [np.concatenate(matrices, axis=0) for matrices in crossnobis_matrices_all_truncate_end]
    crossnobis_matrices_all_truncate_front = [np.concatenate(matrices, axis=0) for matrices in crossnobis_matrices_all_truncate_front]
    full_trial_lengths_all                 = [np.concatenate(lengths, axis=0) for lengths in full_trial_lengths_all]
    neural_speed_all_truncate_end = []
    neural_speed_all_truncate_front = []

    for i_tf, trial_filter in enumerate(trial_filters):

        neural_speed_truncate_end = np.array([np.diag(m, k=superdiagonal_order) / superdiagonal_order for m in crossnobis_matrices_all_truncate_end[i_tf]])
        neural_speed_truncate_front = np.array([np.diag(m, k=superdiagonal_order) / superdiagonal_order for m in crossnobis_matrices_all_truncate_front[i_tf]])

        neural_speed_all_truncate_end.append(neural_speed_truncate_end)
        neural_speed_all_truncate_front.append(neural_speed_truncate_front)

        # analyze_neural_speed(neural_speed_truncate_end, neural_speed_truncate_front)


    analyze_neural_speed_all_conditions(neural_speed_all_truncate_end, neural_speed_all_truncate_front)


    # Steady-phase duration requires per-trial data; skip when per_target
    # (observation unit is targets, not trials).
    if not per_target:
        analyze_neural_speed_phase(
            neural_speed_all_truncate_end,
            full_trial_lengths_all)









def discrete_states_over_time_statistics(
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

    color_palette = trial_filter_colors_by_discrete_state

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
            trial_filter_name_conversion[trial_filter] + ' Trial Count',
            fontsize=5,
        )

        # gather initial-state durations for the stats that follow
        initial_durations = utils_vis.find_initial_zero_durations(discrete_state_matrix_all)
        initial_durations_all.append(initial_durations * time_step)

        axs[i_tf].spines['top'].set_visible(False)  # hide top spine
        axs[i_tf].spines['right'].set_visible(False)  # hide right spine


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
    stat, p_value = mannwhitneyu(initial_durations_all[0],
                                 initial_durations_all[1],
                                 alternative='two-sided')
    print('Fast trial mean:', np.mean(initial_durations_all[0]),
          'Slow trial mean:', np.mean(initial_durations_all[1]))
    print("Mann-Whitney U:", stat, "  P-value:", p_value)
    utils_vis.confidence_interval_95_unpaired(initial_durations_all[0],
                                              initial_durations_all[1])

    plt.tight_layout()

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

    save_path = os.path.join(vis_dir, res_name + '.pdf')
    plt.savefig(save_path, dpi=600, transparent=True, bbox_inches=None)
    plt.close()


def compute_skew_symmetric_ratio(
    session_data_names,
    unit_filter,
    input_unit_filter,
    window_config,
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
):
    """Compute the ratio of skew-symmetric to symmetric components
    of the dynamics matrix A for each discrete state of the model.

    For rSLDS: one ratio per discrete state (K ratios).
    For LDS / rSLDS with K=1: a single ratio.

    Prints results as a table and returns a DataFrame.

    Returns
    -------
    df : pd.DataFrame with columns
         [session, trial_filter, discrete_state, skew_ratio, skew_norm, sym_norm]
    """
    rows = []

    for session_data_name in session_data_names:
        for trial_filter in trial_filters:

            dl = utils_processing.DataLoader(
                data_dir, results_dir, session_data_name,
                unit_filter, input_unit_filter, window_config,
                trial_filter)

            model_results_dir = dl.get_model_result_dir(
                time_offset=None,
                data_format=data_format,
                train_test=train_test_option,
                model_type=model_type,
                dynamics_class=dynamics_class,
                emission_class=emission_class,
                init_type=init_type,
                subspace_type=subspace_type,
                alpha=alpha,
                check_existence=True)

            if model_type in ['LDS']:
                model_save_name = '_'.join(map(str, [
                    'r' + str(random_state),
                    's' + str(n_continuous_states),
                    'i' + str(n_iters)]))
            else:
                model_save_name = '_'.join(map(str, [
                    'r' + str(random_state),
                    's' + str(n_continuous_states),
                    'd' + str(n_discrete_states),
                    'i' + str(n_iters)]))

            model_path = os.path.join(model_results_dir, model_save_name + '.pkl')

            try:
                with open(model_path, 'rb') as f:
                    res = pickle.load(f)
            except FileNotFoundError:
                print(f'Model not found: {model_path}')
                continue

            model = res['model']
            As = model.dynamics.As  # (K, D, D)

            for k in range(As.shape[0]):
                A_k = As[k]
                A_sym  = (A_k + A_k.T) / 2
                A_skew = (A_k - A_k.T) / 2

                sym_norm  = np.linalg.norm(A_sym,  'fro')
                skew_norm = np.linalg.norm(A_skew, 'fro')
                ratio = skew_norm / sym_norm if sym_norm > 0 else np.nan

                rows.append({
                    'session': session_data_name,
                    'trial_filter': trial_filter,
                    'discrete_state': k,
                    'skew_ratio': ratio,
                    'skew_norm': skew_norm,
                    'sym_norm': sym_norm,
                })

    df = pd.DataFrame(rows)
    print('\n=== Skew-symmetric / symmetric ratio of dynamics matrix A ===')
    print(df.to_string(index=False))

    # Save as CSV
    task_name = session_data_names[0].split('_')[-1]
    if len(session_data_names) > 3:
        session_str = str(len(session_data_names)) + '_sessions'
    else:
        session_str = str(session_data_names)

    csv_name = '_'.join(map(str, [x for x in [
        task_name,
        'skew_symmetric_ratio',
        session_str,
        unit_filter,
        window_config,
        train_test_option,
        model_type,
        n_continuous_states,
        n_discrete_states,
    ] if x is not None]))

    csv_path = os.path.join(vis_dir, csv_name + '.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nSaved to {csv_path}')

    # Simple bar chart
    fig, ax = plt.subplots(figsize=(90*mm, 50*mm))

    for i_tf, tf in enumerate(trial_filters):
        df_tf = df[df['trial_filter'] == tf]
        task_name = session_data_names[0].split('_')[-1]
        for k in df_tf['discrete_state'].unique():
            df_k = df_tf[df_tf['discrete_state'] == k]
            mean_ratio = df_k['skew_ratio'].mean()
            se_ratio = df_k['skew_ratio'].std() / np.sqrt(len(df_k)) if len(df_k) > 1 else 0

            x_pos = i_tf * (n_discrete_states + 1) + k
            color = color_palettes[task_name][tf][1]
            ax.bar(x_pos, mean_ratio, yerr=se_ratio,
                   color=color, alpha=0.7, width=0.8,
                   capsize=2, error_kw={'lw': 0.5})

    ax.set_ylabel('||A_skew|| / ||A_sym||', fontsize=7)
    ax.set_xlabel('Discrete state', fontsize=7)
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()

    fig_name = csv_name + '.pdf'
    fig_path = os.path.join(vis_dir, fig_name)
    fig.savefig(fig_path, dpi=600, transparent=True, bbox_inches=None)
    plt.close(fig)

    return df


def steady_phase_target_decoding(
    session_data_names,
    unit_filter,
    window_config,
    trial_filter,
    steady_window_ms=300,
    n_cv_folds=5,
):
    """Decode target direction from steady-phase neural activity.

    For each session, extracts the last `steady_window_ms` of firing rates
    before target acquisition (from the sustained / slow / far condition),
    then runs two decoders:

    1. **Classification** (LDA): cross-validated discrete target prediction,
       returns per-trial correctness (1/0).
    2. **Continuous angle** (Ridge on cos/sin): cross-validated regression
       onto unit-circle representation of target angle, returns per-trial
       absolute angular error in degrees.

    Parameters
    ----------
    session_data_names : list of str
    unit_filter        : str
    window_config      : str
    trial_filter       : str  – typically 'slow' or 'far'
    steady_window_ms   : int  – width of the steady-phase window in ms
    n_cv_folds         : int  – number of cross-validation folds

    Returns
    -------
    all_trial_correct  : np.ndarray  – 1/0 per trial across all sessions
    all_angular_errors : np.ndarray  – absolute angular error (degrees) per trial
    results_df         : pd.DataFrame  – per-session summary
    """

    time_step = float(window_config.split('_s')[1].split('_')[0])
    steady_window_bins = int(steady_window_ms / 1000 / time_step)

    results_rows = []
    all_trial_correct = []
    all_angular_errors = []

    for session_data_name in session_data_names:

        dl = utils_processing.DataLoader(
            data_dir, results_dir, session_data_name,
            unit_filter, None, window_config, trial_filter)
        dl.load_firing_rate_data()

        target_indices = dl.get_target_indices()
        target_positions = dl.get_target_positions()  # (n_trials, 2)
        target_angles = np.arctan2(target_positions[:, 1],
                                   target_positions[:, 0])  # radians
        n_targets = len(np.unique(target_indices))
        chance_accuracy = 1.0 / n_targets
        # Chance angular error: mean of uniform distribution on [0°, 180°]
        chance_angular_error = 90.0

        # Extract steady-phase features: average firing rate in last N bins
        X_list = []
        y_idx_list = []
        y_angle_list = []
        valid_mask = []

        for i, fr in enumerate(dl.firing_rates):
            fr_array = np.array(fr)  # (T_i, N)
            T_i = fr_array.shape[0]

            if T_i < steady_window_bins:
                valid_mask.append(False)
                continue

            steady_fr = fr_array[-steady_window_bins:, :]
            X_list.append(np.mean(steady_fr, axis=0))
            y_idx_list.append(target_indices[i])
            y_angle_list.append(target_angles[i])
            valid_mask.append(True)

        X = np.array(X_list)              # (n_valid_trials, N)
        y = np.array(y_idx_list)          # (n_valid_trials,)
        y_angle = np.array(y_angle_list)  # (n_valid_trials,) radians

        # ── Classification (LDA) ─────────────────────────────────────
        n_classes = len(np.unique(y))
        min_class_count = min(np.bincount(y)[np.bincount(y) > 0])
        effective_folds = min(n_cv_folds, min_class_count)

        if effective_folds < 2:
            print(f'  {session_data_name}: too few trials per target for CV')
            continue

        lda = LinearDiscriminantAnalysis()
        cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=42)
        y_pred = cross_val_predict(lda, X, y, cv=cv)

        trial_correct = (y_pred == y).astype(int)
        all_trial_correct.append(trial_correct)

        mean_acc = np.mean(trial_correct)
        se_acc = np.std(trial_correct) / np.sqrt(len(trial_correct))

        # ── Continuous angle decoding (Ridge on cos/sin) ─────────────
        y_cos_sin = np.column_stack([np.cos(y_angle), np.sin(y_angle)])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        cv_reg = KFold(n_splits=effective_folds, shuffle=True, random_state=42)
        y_cos_sin_pred = cross_val_predict(
            Ridge(alpha=1.0), X_scaled, y_cos_sin, cv=cv_reg)

        # Recover predicted angle and compute absolute angular error
        pred_angle = np.arctan2(y_cos_sin_pred[:, 1], y_cos_sin_pred[:, 0])
        angular_diff = np.abs(np.angle(
            np.exp(1j * pred_angle) / np.exp(1j * y_angle)))
        angular_error_deg = np.degrees(angular_diff)

        all_angular_errors.append(angular_error_deg)

        mean_err = np.mean(angular_error_deg)
        median_err = np.median(angular_error_deg)

        results_rows.append({
            'session': session_data_name,
            'mean_accuracy': mean_acc,
            'se_accuracy': se_acc,
            'mean_angular_error': mean_err,
            'median_angular_error': median_err,
            'chance': chance_accuracy,
            'chance_angular_error': chance_angular_error,
            'n_trials': len(y),
            'n_targets': n_targets,
            'n_folds': effective_folds,
        })

        print(f'  {session_data_name}: accuracy = {mean_acc:.3f} +/- {se_acc:.3f} '
              f'(chance = {chance_accuracy:.3f}), '
              f'angular error = {mean_err:.1f}° (median {median_err:.1f}°), '
              f'n={len(y)} trials')

    results_df = pd.DataFrame(results_rows)

    if len(all_trial_correct) > 0:
        all_trial_correct = np.concatenate(all_trial_correct)
        all_angular_errors = np.concatenate(all_angular_errors)
    else:
        all_trial_correct = np.array([])
        all_angular_errors = np.array([])

    print(f'\n=== {unit_filter} {trial_filter}: '
          f'pooled accuracy = {np.mean(all_trial_correct):.3f}, '
          f'pooled angular error = {np.mean(all_angular_errors):.1f}° '
          f'(n={len(all_trial_correct)} trials) ===')

    return all_trial_correct, all_angular_errors, results_df


def steady_phase_pca_trajectories(
    session_data_name,
    unit_filter,
    window_config,
    trial_filter,
    steady_window_ms=300,
):
    """Plot PCA trajectories of the last `steady_window_ms` of each trial.

    Each trial is a line in PC space, colored by target direction.
    Produces one figure per session with PC1-PC2 and PC1-PC3 panels.
    """

    time_step = float(window_config.split('_s')[1].split('_')[0])
    steady_window_bins = int(steady_window_ms / 1000 / time_step)

    dl = utils_processing.DataLoader(
        data_dir, results_dir, session_data_name,
        unit_filter, None, window_config, trial_filter)
    dl.load_firing_rate_data()

    target_indices = dl.get_target_indices()
    n_targets = len(np.unique(target_indices))

    # Collect steady-phase snippets (variable length OK, but only keep last N bins)
    snippets = []  # list of (T_clip, N) arrays
    snippet_targets = []

    for i, fr in enumerate(dl.firing_rates):
        fr_array = np.array(fr)
        T_i = fr_array.shape[0]
        if T_i < steady_window_bins:
            continue
        snippets.append(fr_array[-steady_window_bins:, :])
        snippet_targets.append(target_indices[i])

    snippet_targets = np.array(snippet_targets)

    # Fit PCA on concatenated steady-phase data
    all_steady = np.concatenate(snippets, axis=0)  # (total_bins, N)
    pca = PCA(n_components=min(2, all_steady.shape[1]))
    pca.fit(all_steady)

    # Project each trial snippet
    projected = [pca.transform(s) for s in snippets]  # list of (T_clip, 2)

    target_palette = sns.color_palette('hls', n_targets)
    unique_targets = np.sort(np.unique(snippet_targets))
    color_map = {tgt: target_palette[i] for i, tgt in enumerate(unique_targets)}

    fig, ax = plt.subplots(figsize=(45 * mm, 45 * mm))

    for proj, tgt in zip(projected, snippet_targets):
        ax.plot(proj[:, 0], proj[:, 1],
                color=color_map[tgt], alpha=0.3, lw=0.4)

    # Plot target-average trajectories on top
    for tgt in unique_targets:
        mask = snippet_targets == tgt
        tgt_projs = [projected[i] for i in range(len(projected)) if mask[i]]
        avg_proj = np.mean(tgt_projs, axis=0)

        ax.plot(avg_proj[:, 0], avg_proj[:, 1],
                color=color_map[tgt], alpha=1.0, lw=1.5)
        # Mark end point
        ax.plot(avg_proj[-1, 0], avg_proj[-1, 1],
                'o', color=color_map[tgt], markersize=3, zorder=5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=5)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=5)
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()

    task_name = session_data_name.split('_')[-1]
    fig_name = '_'.join([
        task_name, 'steady_phase_PCA_trajectories',
        session_data_name, unit_filter, trial_filter,
        f'{steady_window_ms}ms',
    ])
    fig_path = os.path.join(vis_dir, fig_name + '.pdf')
    fig.savefig(fig_path, dpi=600, transparent=True, bbox_inches=None)
    plt.close(fig)
    print(f'  Saved PCA trajectories: {fig_path}')


def plot_steady_phase_decoding_summary(
    all_results,
    steady_window_ms=300,
):
    """Summary plots of steady-phase decoding across subjects and tasks.

    Generates two figures:
    1. Classification accuracy bar plot (original).
    2. Angular error box plot (continuous angle decoding).

    Parameters
    ----------
    all_results : list of dict
        Each dict has keys: 'label', 'trial_correct' (0/1 array),
        'angular_errors' (degrees array), 'chance', 'chance_angular_error'.
    steady_window_ms : int
    """

    # Task short name -> light color from vis_config color_palettes
    task_bar_colors = {
        'CO':  theme_orange_light,   # CenterStart slow
        'RG':  theme_coral_light,    # RadialGrid far
        'COI': theme_yellow_light,   # CenterStartInterleave slow
    }

    labels = [r['label'] for r in all_results]
    means = [np.mean(r['trial_correct']) for r in all_results]
    sems = [np.std(r['trial_correct']) / np.sqrt(len(r['trial_correct']))
            for r in all_results]
    chances = [r['chance'] for r in all_results]

    # Assign bar color based on task (last token in label)
    bar_colors = [task_bar_colors.get(lab.split()[-1], 'steelblue')
                  for lab in labels]

    # ==================================================================
    # Figure 1: Classification accuracy (bar plot)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(60 * mm, 45 * mm))

    x_pos = np.arange(len(labels))
    ax.bar(x_pos, means, yerr=sems,
           color=bar_colors, alpha=0.9, capsize=1.5,
           error_kw={'lw': 0.3, 'capthick': 0.3}, edgecolor='none')

    # Chance line
    max_chance = max(chances)
    ax.axhline(y=max_chance, color='black', linestyle='--', lw=size_line_thin,
               label=f'Chance (1/{int(round(1/max_chance))})')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=5, rotation=30, ha='right')
    ax.set_ylabel('Decoding accuracy', fontsize=7)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=5, frameon=False)

    # Annotate n trials
    for i, r in enumerate(all_results):
        ax.text(i, means[i] + sems[i] + 0.02,
                f'n={len(r["trial_correct"])}',
                ha='center', va='bottom', fontsize=5)

    plt.tight_layout()

    fig_name = f'steady_phase_decoding_summary_{steady_window_ms}ms'
    fig_path = os.path.join(vis_dir, fig_name + '.pdf')
    fig.savefig(fig_path, dpi=600, transparent=True, bbox_inches=None)
    plt.close(fig)
    print(f'Saved classification plot: {fig_path}')

    # ==================================================================
    # Figure 2: Angular error (box plot)
    # ==================================================================
    fig2, ax2 = plt.subplots(figsize=(60 * mm, 45 * mm))

    angular_data = [r['angular_errors'] for r in all_results]
    bp = ax2.boxplot(
        angular_data, positions=x_pos, widths=0.5,
        whis=[0, 100], showfliers=False,
        patch_artist=True, manage_ticks=False)

    for i, box in enumerate(bp['boxes']):
        box.set(facecolor=bar_colors[i], linewidth=0, alpha=0.8)
    for line in bp['whiskers'] + bp['caps']:
        line.set(color='black', linewidth=0.25)
    for line in bp['medians']:
        line.set(color='black', linewidth=0.5)

    # Chance line (mean angular error for uniform guessing = 90°)
    max_chance_err = max(r.get('chance_angular_error', 90.0)
                         for r in all_results)
    ax2.axhline(y=max_chance_err, color='black', linestyle='--',
                lw=size_line_thin)
    # ax2.text(0.98, 0.97, 'Chance (90°)', transform=ax2.transAxes,
    #          ha='right', va='top', fontsize=5)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=5, rotation=30, ha='right')
    ax2.set_ylabel('Angular error (°)', fontsize=7)
    ax2.set_ylim(-10, 190)
    ax2.set_yticks([0, 45, 90, 135, 180])
    ax2.tick_params(axis='both', which='major', labelsize=5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Annotate n trials
    for i, r in enumerate(all_results):
        median_err = np.median(r['angular_errors'])
        ax2.text(i, np.max(r['angular_errors']) + 3,
                 f'n={len(r["angular_errors"])}',
                 ha='center', va='bottom', fontsize=5)

    plt.tight_layout()

    fig_name2 = f'steady_phase_angular_error_summary_{steady_window_ms}ms'
    fig_path2 = os.path.join(vis_dir, fig_name2 + '.pdf')
    fig2.savefig(fig_path2, dpi=600, transparent=True, bbox_inches=None)
    plt.close(fig2)
    print(f'Saved angular error plot: {fig_path2}')


def steady_phase_progress_to_target(
    session_data_name,
    unit_filter,
    window_config,
    trial_filter,
    steady_window_ms=300,
    alpha_ridge=1.0,
):
    """Plot "progress to target" neural axis trajectory in the last N ms.

    Fits a ridge regression from time-binned neural activity to cursor error
    (target_pos - cursor_pos).  The learned weight vector defines a
    task-relevant "progress-to-target" axis.  Each trial's last
    `steady_window_ms` of neural activity is projected onto this axis and
    plotted as a 1-D trajectory over time, colored by target.

    Parameters
    ----------
    session_data_name : str
    unit_filter       : str
    window_config     : str
    trial_filter      : str
    steady_window_ms  : int
    alpha_ridge       : float  – Ridge regularisation strength
    """

    time_step = float(window_config.split('_s')[1].split('_')[0])
    steady_window_bins = int(steady_window_ms / 1000 / time_step)

    dl = utils_processing.DataLoader(
        data_dir, results_dir, session_data_name,
        unit_filter, None, window_config, trial_filter)
    dl.load_firing_rate_data()
    dl.load_cursor_data()

    target_positions = dl.get_target_positions()   # (n_trials, 2)
    target_indices = dl.get_target_indices()
    n_targets = len(np.unique(target_indices))

    # Collect per-bin neural activity and cursor error in steady phase
    from scipy.interpolate import interp1d

    X_bins = []   # neural activity at each time bin
    y_dist = []   # scalar distance to target at each time bin
    trial_ids = []
    trial_targets = []

    for i_trial, fr in enumerate(dl.firing_rates):
        fr_array = np.array(fr)           # (T_i, N)
        T_i = fr_array.shape[0]

        if T_i < steady_window_bins:
            continue

        # Interpolate cursor position to firing rate time bins
        times_fr = np.array(dl.times[i_trial])
        cursor_pos_xr = dl.cursor_pos.iloc[i_trial]   # xarray (time, 2)
        times_cursor = np.array(cursor_pos_xr.time)
        f_x = interp1d(times_cursor, cursor_pos_xr[:, 0],
                        kind='linear', fill_value='extrapolate')
        f_y = interp1d(times_cursor, cursor_pos_xr[:, 1],
                        kind='linear', fill_value='extrapolate')
        cursor_xy = np.column_stack([f_x(times_fr), f_y(times_fr)])  # (T_i, 2)

        fr_steady = fr_array[-steady_window_bins:, :]
        cursor_steady = cursor_xy[-steady_window_bins:, :]
        target_pos = target_positions[i_trial]  # (2,)
        # Scalar distance to target (direction-invariant)
        dist = np.linalg.norm(target_pos[None, :] - cursor_steady, axis=1)  # (T_clip,)

        X_bins.append(fr_steady)
        y_dist.append(dist)
        trial_ids.append(i_trial)
        trial_targets.append(target_indices[i_trial])

    if len(X_bins) == 0:
        print(f'  {session_data_name}: no valid trials for progress-to-target')
        return

    trial_targets = np.array(trial_targets)

    # Fit ridge regression: neural activity -> scalar distance to target
    X_all = np.concatenate(X_bins, axis=0)     # (total_bins, N)
    y_all = np.concatenate(y_dist, axis=0)     # (total_bins,)

    ridge = Ridge(alpha=alpha_ridge)
    ridge.fit(X_all, y_all)

    # The weight vector IS the progress-to-target axis (1D output)
    progress_axis = ridge.coef_  # (N,)
    progress_axis = progress_axis / np.linalg.norm(progress_axis)

    # Project each trial's steady-phase activity onto the axis
    projections = []
    for fr_steady in X_bins:
        proj = fr_steady @ progress_axis  # (T_clip,)
        projections.append(proj)

    time_ms = np.arange(-steady_window_bins, 0) * time_step * 1000  # ms before end

    target_palette = sns.color_palette('hls', n_targets)
    unique_targets = np.sort(np.unique(trial_targets))
    color_map = {tgt: target_palette[j] for j, tgt in enumerate(unique_targets)}

    fig, ax = plt.subplots(figsize=(55 * mm, 45 * mm))

    # Individual trial trajectories (thin)
    for proj, tgt in zip(projections, trial_targets):
        ax.plot(time_ms, proj, color=color_map[tgt], alpha=0.15, lw=0.3)

    # Target-average trajectories (thick)
    for tgt in unique_targets:
        mask = trial_targets == tgt
        tgt_projs = [projections[i] for i in range(len(projections)) if mask[i]]
        avg_proj = np.mean(tgt_projs, axis=0)
        ax.plot(time_ms, avg_proj, color=color_map[tgt], lw=1.5, alpha=1.0)

    ax.set_xlabel('Time before trial end (ms)', fontsize=5)
    ax.set_ylabel('Progress-to-target axis (a.u.)', fontsize=5)
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Report ridge R2
    y_pred = ridge.predict(X_all)
    ss_res = np.sum((y_all - y_pred) ** 2)
    ss_tot = np.sum((y_all - np.mean(y_all, axis=0)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    ax.set_title(f'Ridge R\u00b2 = {r2:.3f}', fontsize=5)

    fig.tight_layout()

    task_name = session_data_name.split('_')[-1]
    fig_name = '_'.join([
        task_name, 'progress_to_target',
        session_data_name, unit_filter, trial_filter,
        f'{steady_window_ms}ms',
    ])
    fig_path = os.path.join(vis_dir, fig_name + '.pdf')
    fig.savefig(fig_path, dpi=600, transparent=True, bbox_inches=None)
    plt.close(fig)
    print(f'  Saved progress-to-target: {fig_path}')


def crossnobis_robustness_check(
    session_data_names,
    unit_filter,
    window_config,
    time_step,
    data_format,
    trial_filters,
    reaction_time,
    visual_delay_times,
    peak_times,
    truncate_percentile=90,
    pre_start_time_buffer=0.2,
    post_reach_time_buffer=0.5,
    significance_alpha=0.05,
    correction_method='fdr_bh',
    superdiagonal_order=1,
    per_target=False,
    recompute=False,
    recompute_reduced=False,
):
    """Recompute crossnobis after removing cross-condition discrepant neurons.

    recompute : bool
        If True, ignore all caches and recompute from scratch.
    recompute_reduced : bool
        If True, recompute discrepant-neuron masks and reduced crossnobis
        even when a cached pkl exists.  Original (no neuron removal) crossnobis
        matrices are still loaded from the pkl when available.


    Generates three figure types:
      1. Superdiagonal comparison (two-panel: truncate_end + truncate_front),
         overlaying original and neuron-removed traces per condition.
      2. Matrix heatmap comparison (original vs reduced per condition).
      3. Epoch-wise bar plots summarizing the effect of neuron removal.

    All crossnobis results are saved to a pickle file for reuse.
    Set ``recompute=False`` (default) to load cached results when available.
    """
    import vis_paper
    import matplotlib.ticker as ticker

    pre_start_idx  = int(pre_start_time_buffer / time_step)
    post_reach_idx = int(post_reach_time_buffer / time_step)

    task_name = session_data_names[0].split('_')[-1]
    color_palettes_ = color_palettes[task_name]

    # Use first session's values for plotting markers
    visual_delay_time = (visual_delay_times[0]
                         if isinstance(visual_delay_times, list)
                         else visual_delay_times)
    peak_time = (peak_times[0]
                 if isinstance(peak_times, list)
                 else peak_times)

    # ── File‑name base (shared by pkl + all figures) ─────────────────
    if len(session_data_names) > 3:
        session_str = str(len(session_data_names)) + '_sessions'
    else:
        session_str = str(session_data_names)

    base_name = '_'.join(str(x) for x in [
        task_name, 'crossnobis_robustness', session_str,
        unit_filter, window_config,
    ] if x is not None)

    pkl_path = os.path.join(vis_dir, base_name + '_results.pkl')

    # ── Formats needed ───────────────────────────────────────────────
    # Bar plot uses truncate_end only.
    formats_needed = ['truncate_end']

    # ==================================================================
    # COMPUTE (or load cached results)
    # ==================================================================
    pkl_exists = not recompute and os.path.exists(pkl_path)

    if pkl_exists and not recompute_reduced:
        # Load everything from pkl (original + reduced + discrepant_masks)
        print(f'  Loading cached results: {pkl_path}')
        with open(pkl_path, 'rb') as f:
            save_data = pickle.load(f)

        crossnobis = {}
        for trial_filter in trial_filters:
            crossnobis[trial_filter] = {}
            for fmt in formats_needed:
                stored = save_data[trial_filter][fmt]
                crossnobis[trial_filter][fmt] = {
                    'original':   stored['original_matrices'],
                    'reduced':    stored['reduced_matrices'],
                    'min_length': stored['min_length'],
                }
        discrepant_masks = {
            k: np.array(v) for k, v in save_data['discrepant_masks'].items()
        }
    else:
        # ── 1. Identify discrepant neurons per session ───────────────
        discrepant_masks = {}
        for i_ses, session_data_name in enumerate(session_data_names):
            vdt = (visual_delay_times[i_ses]
                   if isinstance(visual_delay_times, list)
                   else visual_delay_times)
            pkt = (peak_times[i_ses]
                   if isinstance(peak_times, list)
                   else peak_times)

            sig_matrix = vis_paper._neuron_cross_condition_discrepancy_one_session(
                session_data_name, unit_filter, time_step, window_config,
                data_format, trial_filters, reaction_time, vdt, pkt,
                truncate_percentile=truncate_percentile,
                significance_alpha=significance_alpha,
                correction_method=correction_method,
                pre_start_time_buffer=pre_start_time_buffer,
                post_reach_time_buffer=post_reach_time_buffer)

            discrepant_mask = sig_matrix.any(axis=0).any(axis=1)
            discrepant_masks[session_data_name] = discrepant_mask
            n_total = len(discrepant_mask)
            n_removed = discrepant_mask.sum()
            print(f'  {session_data_name}: '
                  f'removing {n_removed}/{n_total} discrepant neurons')

        # ── 2. Load originals (cached) & compute reduced ────────────
        crossnobis = {
            tf: {fmt: {'original': [], 'reduced': [], 'min_length': np.inf}
                 for fmt in formats_needed}
            for tf in trial_filters
        }

        for session_data_name in session_data_names:
            keep_mask = ~discrepant_masks[session_data_name]

            for trial_filter in trial_filters:
                for fmt in formats_needed:
                    # Original: always load per-session (never from
                    # concatenated pkl, which may have stale counts)
                    stack_orig, dl_orig = \
                        vis_paper._load_or_compute_within_crossnobis(
                            session_data_name, unit_filter,
                            window_config, trial_filter,
                            fmt, time_step, truncate_percentile,
                            pre_start_idx, post_reach_idx,
                            per_target=per_target,
                            per_target_stack=per_target,
                            recompute=recompute)

                    # Reduced: skip cache when recompute_reduced=True
                    reduced_cache = \
                        vis_paper._within_crossnobis_cache_path(
                            session_data_name, unit_filter,
                            window_config, trial_filter,
                            per_target=per_target,
                            suffix='_reduced')
                    fmt_key = f'crossnobis_matrices_{fmt}'
                    stack_red = None

                    if not recompute and not recompute_reduced \
                            and os.path.exists(reduced_cache):
                        with open(reduced_cache, 'rb') as f:
                            cached_red = pickle.load(f)
                        if fmt_key in cached_red:
                            stack_red = cached_red[fmt_key]
                            print(f'    Loaded cached reduced '
                                  f'{fmt}: {reduced_cache}')

                    if stack_red is None:
                        # Compute reduced from neuron-subsetted data
                        dl = utils_processing.DataLoader(
                            data_dir, results_dir,
                            session_data_name, unit_filter,
                            None, window_config, trial_filter)
                        dl.load_firing_rate_data()

                        fr, *_ = dl.reformat_firing_rate_data(
                            fmt,
                            index_buffer=post_reach_idx,
                            trial_length_filter_percentile=
                                truncate_percentile)

                        if fmt == 'truncate_end':
                            fr = fr[:, :-post_reach_idx, :]
                        elif fmt == 'truncate_front':
                            fr = fr[:, pre_start_idx:, :]

                        fr_reduced = fr[:, :, keep_mask]

                        target_angles = (dl.get_target_angles()
                                         if per_target else None)

                        stack_red = \
                            vis_paper._compute_within_crossnobis_from_data(
                                fr_reduced, time_step,
                                per_target=per_target,
                                target_angles=target_angles)

                        # Save reduced to its own cache file
                        if os.path.exists(reduced_cache):
                            with open(reduced_cache, 'rb') as f:
                                cached_red = pickle.load(f)
                        else:
                            cached_red = {}
                        cached_red[fmt_key] = stack_red
                        os.makedirs(os.path.dirname(reduced_cache),
                                    exist_ok=True)
                        with open(reduced_cache, 'wb') as f:
                            pickle.dump(cached_red, f)
                        print(f'    Saved reduced {fmt}: '
                              f'{reduced_cache}')

                    entry = crossnobis[trial_filter][fmt]
                    entry['original'].append(stack_orig)
                    entry['reduced'].append(stack_red)
                    entry['min_length'] = min(
                        entry['min_length'], stack_orig.shape[-1])

        # ── 3. Truncate to common length & concatenate ───────────────
        for trial_filter in trial_filters:
            for fmt in formats_needed:
                entry = crossnobis[trial_filter][fmt]
                ml = int(entry['min_length'])
                for key in ['original', 'reduced']:
                    for i in range(len(entry[key])):
                        s = entry[key][i]
                        if fmt == 'truncate_front':
                            entry[key][i] = s[:, -ml:, -ml:]
                        else:
                            entry[key][i] = s[:, :ml, :ml]
                    entry[key] = np.concatenate(entry[key], axis=0)

        # ── 4. Save results ──────────────────────────────────────────
        save_data = {
            'session_data_names': session_data_names,
            'unit_filter': unit_filter,
            'window_config': window_config,
            'time_step': time_step,
            'data_format': data_format,
            'trial_filters': trial_filters,
            'discrepant_masks': {
                k: v.tolist() for k, v in discrepant_masks.items()},
            'pre_start_idx': pre_start_idx,
            'post_reach_idx': post_reach_idx,
            'superdiagonal_order': superdiagonal_order,
            'visual_delay_time': visual_delay_time,
            'peak_time': peak_time,
            'per_target': per_target,
        }
        for trial_filter in trial_filters:
            save_data[trial_filter] = {}
            for fmt in formats_needed:
                entry = crossnobis[trial_filter][fmt]
                save_data[trial_filter][fmt] = {
                    'original_matrices': entry['original'],
                    'reduced_matrices':  entry['reduced'],
                    'min_length':        int(entry['min_length']),
                }

        with open(pkl_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f'  Saved results: {pkl_path}')

    # ==================================================================
    # Print neuron removal summary
    # ==================================================================
    for sdn, mask in discrepant_masks.items():
        mask = np.asarray(mask)
        print(f'  {sdn}: {mask.sum()}/{len(mask)} discrepant neurons')

    # ==================================================================
    # Epoch‑wise bar plots with statistics
    # ==================================================================
    from scipy.stats import wilcoxon

    fmt_bar = 'truncate_end'
    vd_bins = int(visual_delay_time / time_step)
    pk_bins = int(peak_time / time_step)

    epoch_names = ['Visual delay', 'Before peak', 'After peak']
    # Gray shades matching dark/mid/light intensity
    grays = ['#4D4D4D', '#808080', '#B3B3B3']
    bar_data = []

    for trial_filter in trial_filters:
        entry = crossnobis[trial_filter][fmt_bar]

        for version in ['original', 'reduced']:
            mats = entry[version]
            diag_stack = np.array([
                np.diag(m, k=superdiagonal_order) / superdiagonal_order
                for m in mats])
            n_diag = diag_stack.shape[1]

            epoch_slices = {
                'Visual delay': (pre_start_idx,
                                 pre_start_idx + vd_bins),
                'Before peak':  (pre_start_idx + vd_bins,
                                 pre_start_idx + pk_bins),
                'After peak':   (pre_start_idx + pk_bins,
                                 n_diag),
            }

            for epoch_name in epoch_names:
                s, e = epoch_slices[epoch_name]
                s = max(0, s)
                e = min(n_diag, e)
                if e <= s:
                    continue
                epoch_vals = np.mean(diag_stack[:, s:e], axis=1)
                for val in epoch_vals:
                    bar_data.append({
                        'trial_filter': trial_filter_name_conversion.get(
                            trial_filter, trial_filter),
                        'trial_filter_raw': trial_filter,
                        'version': version,
                        'epoch': epoch_name,
                        'value': val,
                    })

    df_bar = pd.DataFrame(bar_data)
    n_tf = len(trial_filters)

    fig_bar, axs_bar = plt.subplots(
        1, n_tf, figsize=(80*mm, 45*mm), sharey=True)
    if n_tf == 1:
        axs_bar = [axs_bar]

    for i_tf, trial_filter in enumerate(trial_filters):
        ax = axs_bar[i_tf]
        tf_name = trial_filter_name_conversion.get(
            trial_filter, trial_filter)
        df_tf = df_bar[df_bar['trial_filter_raw'] == trial_filter]
        cond_colors = color_palettes_[trial_filter]  # [light, mid, dark]

        # Per-epoch colors: dark, mid, light for each epoch
        epoch_color_map = {
            'Visual delay': {'original': grays[0],
                             'reduced':  cond_colors[2]},
            'Before peak':  {'original': grays[1],
                             'reduced':  cond_colors[1]},
            'After peak':   {'original': grays[2],
                             'reduced':  cond_colors[0]},
        }

        n_epochs = len(epoch_names)
        bar_width = 0.3
        x_positions = np.arange(n_epochs)

        # Draw bars and error bars; collect bracket info
        bracket_info = []
        for i_ep, epoch_name in enumerate(epoch_names):
            df_ep = df_tf[df_tf['epoch'] == epoch_name]
            bar_tops = []
            for j_v, version in enumerate(['original', 'reduced']):
                df_v = df_ep[df_ep['version'] == version]
                vals = df_v['value'].values
                x = x_positions[i_ep] + (j_v - 0.5) * bar_width
                color = epoch_color_map[epoch_name][version]
                # Box plot (whiskers = min/max)
                bp = ax.boxplot(
                    vals, positions=[x], widths=bar_width * 0.8,
                    whis=[0, 100], showfliers=False,
                    patch_artist=True, manage_ticks=False)
                bp['boxes'][0].set(facecolor=color, linewidth=0,
                                   alpha=0.8)
                for line in bp['whiskers'] + bp['caps']:
                    line.set(color='black', linewidth=0.25)
                for line in bp['medians']:
                    line.set(color='black', linewidth=0.5)
                bar_tops.append(np.max(vals))

            # Paired Wilcoxon signed-rank test with rank-biserial r
            orig_vals = df_ep[
                df_ep['version'] == 'original']['value'].values
            red_vals = df_ep[
                df_ep['version'] == 'reduced']['value'].values
            n_pair = min(len(orig_vals), len(red_vals))
            if n_pair >= 5:
                stat, p_val = wilcoxon(orig_vals[:n_pair],
                                       red_vals[:n_pair])
                # Matched-pairs rank-biserial correlation (effect size)
                # r = 1 - (2T)/(n(n+1)/2) where T is the Wilcoxon
                # statistic (smaller rank sum) and n is the number of
                # non-zero differences. This is the standard effect size
                # for the Wilcoxon signed-rank test (Kerby 2014).
                diffs = orig_vals[:n_pair] - red_vals[:n_pair]
                diffs_nz = diffs[diffs != 0]
                n_nz = len(diffs_nz)
                if n_nz > 0:
                    rank_sum_total = n_nz * (n_nz + 1) / 2
                    r_effect = 1 - (2 * stat) / rank_sum_total
                else:
                    r_effect = 0.0

                if p_val < 0.001:
                    sig_label = '***'
                elif p_val < 0.01:
                    sig_label = '**'
                elif p_val < 0.05:
                    sig_label = '*'
                else:
                    sig_label = 'ns'
                # Percentage change: (reduced - original) / original * 100
                mean_orig = np.mean(orig_vals[:n_pair])
                mean_red = np.mean(red_vals[:n_pair])
                if mean_orig != 0:
                    pct_change = (mean_red - mean_orig) / abs(mean_orig) * 100
                else:
                    pct_change = float('nan')

                bracket_info.append((i_ep, max(bar_tops), sig_label, r_effect))
                print(f'  {tf_name} | {epoch_name}: '
                      f'W={stat:.1f}, p={p_val:.4g}, r={r_effect:.3f}, '
                      f'{sig_label} (n={n_pair} paired observations), '
                      f'pct_change={pct_change:+.2f}%')
            else:
                print(f'  {tf_name} | {epoch_name}: '
                      f'too few observations for test (n={n_pair})')

        y_min = -2500
        y_max = 22000
        ax.set_ylim(bottom=y_min, top=y_max)
        tip_h = 0.015 * y_max
        for i_ep, y_top, sig_label, r_effect in bracket_info:
            bracket_h = y_top + 1000
            x_left = x_positions[i_ep] - 0.5 * bar_width
            x_right = x_positions[i_ep] + 0.5 * bar_width
            ax.plot([x_left, x_left, x_right, x_right],
                    [bracket_h - tip_h, bracket_h,
                     bracket_h, bracket_h - tip_h],
                    color='black', linewidth=0.5, clip_on=False)
            if sig_label == 'ns':
                label_text = 'ns'
            else:
                label_text = f"{sig_label}\nr={r_effect:.2f}"
            ax.text((x_left + x_right) / 2, bracket_h + 250,
                    label_text, ha='center', va='bottom',
                    fontsize=5, clip_on=False)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(epoch_names)
        # ax.set_title(tf_name, fontsize=6)
        ax.set_xlabel('')
        if i_tf == 0:
            ax.set_ylabel(
                r'Mean neural speed (($\Delta$Hz / s)$^2$)', fontsize=5)
        ax.tick_params(axis='both', which='major', labelsize=5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        sf_bar = ticker.ScalarFormatter(useMathText=True)
        sf_bar.set_powerlimits((3, 3))
        ax.yaxis.set_major_formatter(sf_bar)

    fig_bar.tight_layout()
    fig_bar.savefig(os.path.join(vis_dir, base_name + '_barplot.pdf'),
                    dpi=600, transparent=True, bbox_inches=None)
    plt.close(fig_bar)
    print(f'  Saved bar plot: {base_name}_barplot.pdf')


def peak_time_significance(
    session_info,
    window_config='gt_-0.2_fct_0.5_s0.001_gaussian_0.03_10',
    features=None):
    """
    Simplified peak-time significance test.
    For each task group in session_info, tests whether peak statistics differ
    between the two trial-filter conditions (fast/slow or near/far).

    Loads pre-computed crossnobis superdiagonal peak stats pkl files
    (same format as statistical_tests_algorithmic.construct_df).

    Produces a summary table (printed + CSV) and a violin plot.
    """
    if features is None:
        features = ['peak_time', 'peak_onset_time', 'peak_duration', 'peak_value']

    from scipy.stats import mannwhitneyu, wilcoxon
    from prettytable import PrettyTable
    import matplotlib.collections as mcoll

    # ── 1. Load all peak stats into a single DataFrame ──
    df_list = []
    for session_key, info in session_info.items():
        sessions      = info['session_data_names']
        unit_filters_ = info['unit_filters']
        trial_filters = info['trial_filters']
        task          = info.get('task', session_key)
        subject       = info.get('subject', session_key)
        uf_short      = info.get('unit_filters_short', unit_filters_)

        for session in sessions:
            session_vis_dir = os.path.join(vis_dir, session)
            for uf, ufs in zip(unit_filters_, uf_short):
                save_name = '_'.join([
                    session,
                    'neural_speed_split',
                    uf,
                    window_config,
                ])
                save_name += '_superdiagonal1_nstd1.5'
                pkl_file = os.path.join(session_vis_dir, save_name + '_stats.pkl')

                if not os.path.exists(pkl_file):
                    print(f'  [SKIP] {pkl_file} not found')
                    continue

                with open(pkl_file, 'rb') as f:
                    res = pickle.load(f)

                for tf in trial_filters:
                    if tf not in res:
                        continue
                    temp = pd.DataFrame(res[tf])
                    temp['session'] = session
                    temp['unit_filter'] = uf
                    temp['unit_filter_short'] = ufs
                    temp['trial_filter'] = tf
                    temp['task'] = task
                    temp['subject'] = subject
                    temp['group'] = f'{subject} {ufs} {info.get("task_short", task)}'
                    df_list.append(temp)

    if not df_list:
        print('No peak stats data found.')
        return

    df = pd.concat(df_list, ignore_index=True)

    # Filter out outlier peaks
    df = df[(df['peak_onset_time'].between(0.1, 1.0)) &
            (df['peak_time'].between(0.1, 1.0))]

    # ── 2. Build within-group pairs (condition1 vs condition2) ──
    groups = df['group'].unique()
    pairs = []
    for g in groups:
        conditions = df[df['group'] == g]['trial_filter'].unique()
        if len(conditions) == 2:
            pairs.append(((g, conditions[0]), (g, conditions[1])))

    # ── 3. Run significance tests ──
    for feature in features:
        print(f'\n{"="*60}')
        print(f'  Feature: {feature}')
        print(f'{"="*60}')

        table = PrettyTable()
        table.field_names = [
            'Comparison', 'N1', 'N2', 'Test',
            'p-value', 'Sig', 'Mean Diff', '95% CI'
        ]

        n_comparisons = len(pairs)

        for (g1, c1), (g2, c2) in pairs:
            arr1 = df[(df['group'] == g1) & (df['trial_filter'] == c1)][feature].dropna().values
            arr2 = df[(df['group'] == g2) & (df['trial_filter'] == c2)][feature].dropna().values

            if len(arr1) < 2 or len(arr2) < 2:
                continue

            # Same-length → paired Wilcoxon; otherwise Mann-Whitney
            if len(arr1) == len(arr2):
                stat, p = wilcoxon(arr1, arr2)
                test_name = 'Wilcoxon'
                diffs = arr1 - arr2
                md = np.mean(diffs)
                se = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
                tc = t.ppf(0.975, len(diffs) - 1)
                ci_lo, ci_hi = md - tc * se, md + tc * se
            else:
                stat, p = mannwhitneyu(arr1, arr2)
                test_name = 'Mann-Whitney'
                md = np.mean(arr1) - np.mean(arr2)
                v1, v2 = np.var(arr1, ddof=1), np.var(arr2, ddof=1)
                n1, n2 = len(arr1), len(arr2)
                se = np.sqrt(v1 / n1 + v2 / n2)
                df_eff = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
                tc = t.ppf(0.975, df_eff)
                ci_lo, ci_hi = md - tc * se, md + tc * se

            # Bonferroni-corrected annotation
            p_corr = p * n_comparisons
            if p_corr <= 1e-4:
                sig = '****'
            elif p_corr <= 1e-3:
                sig = '***'
            elif p_corr <= 1e-2:
                sig = '**'
            elif p_corr <= 5e-2:
                sig = '*'
            else:
                sig = 'ns'

            table.add_row([
                f'{g1} {c1} vs {c2}',
                len(arr1), len(arr2), test_name,
                f'{p:.3g}', sig,
                f'{md:.4f}', f'[{ci_lo:.4f}, {ci_hi:.4f}]'
            ])

        print(table)

    # ── 4. Violin plot ──
    # Rename conditions for display
    df_plot = df.copy()
    rename_map = {'fast': 'ballistic', 'slow': 'sustained'}
    df_plot['trial_filter'] = df_plot['trial_filter'].replace(rename_map)

    # One subplot per task
    tasks = df_plot['task'].unique()
    n_tasks = len(tasks)

    for feature in features:
        fig, axes = plt.subplots(
            ncols=n_tasks, sharey=True,
            figsize=(35 * mm * n_tasks, 40 * mm),
            squeeze=False)
        axes = axes.ravel()

        for i, (task, ax) in enumerate(zip(tasks, axes)):
            df_task = df_plot[df_plot['task'] == task]
            conditions = df_task['trial_filter'].unique()

            palette = {}
            for c in conditions:
                if c == 'ballistic':
                    palette[c] = '#87CCE6'
                elif c == 'sustained':
                    palette[c] = '#FDA058'
                elif c == 'near':
                    palette[c] = '#78CCC2'
                elif c == 'far':
                    palette[c] = '#F79C9C'
                else:
                    palette[c] = '#AAAAAA'

            sns.violinplot(
                ax=ax, data=df_task,
                x='group', y=feature,
                hue='trial_filter', split=True,
                cut=0, inner='quartile',
                palette=palette,
                linewidth=0.25, saturation=1, alpha=0.8)

            for art in ax.findobj(mcoll.PolyCollection):
                art.set_edgecolor('none')
            for l in ax.lines[1::3]:
                l.set_linestyle('-')
            for l in ax.lines[::3]:
                l.set_linestyle('--')
            for l in ax.lines[2::3]:
                l.set_linestyle('--')

            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=5)
            ax.set_title(task, fontsize=6)
            ax.set_xlabel(None)
            sns.despine(ax=ax, top=True, right=True,
                        left=(i > 0), bottom=False)
            if i > 0:
                ax.set_ylabel(None)
                ax.tick_params(axis='y', left=False, labelleft=False)
            else:
                label_map = {
                    'peak_time': 'Peak Time (s)',
                    'peak_onset_time': 'Peak Onset Time (s)',
                    'peak_duration': 'Peak Duration (s)',
                    'peak_value': 'Peak Magnitude',
                }
                ax.set_ylabel(label_map.get(feature, feature), fontsize=5)

            ax.tick_params(axis='both', labelsize=5)
            if ax.get_legend() is not None:
                ax.legend_.remove()

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.05)

        fig_name = f'peak_significance_{feature}.pdf'
        fig_path = os.path.join(vis_dir, fig_name)
        fig.savefig(fig_path, dpi=600, transparent=True, bbox_inches=None)
        plt.close(fig)
        print(f'Saved: {fig_path}')

    # ── 5. Save summary CSV ──
    csv_path = os.path.join(vis_dir, 'peak_significance_summary.csv')
    summary_rows = []
    for (g1, c1), (g2, c2) in pairs:
        row = {'group': g1, 'cond1': c1, 'cond2': c2}
        for feature in features:
            arr1 = df[(df['group'] == g1) & (df['trial_filter'] == c1)][feature].dropna().values
            arr2 = df[(df['group'] == g2) & (df['trial_filter'] == c2)][feature].dropna().values
            row[f'{feature}_mean_cond1'] = np.mean(arr1) if len(arr1) else np.nan
            row[f'{feature}_mean_cond2'] = np.mean(arr2) if len(arr2) else np.nan
            row[f'{feature}_diff'] = row[f'{feature}_mean_cond1'] - row[f'{feature}_mean_cond2']
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f'Saved: {csv_path}')


def lds_vs_rslds_goodness_of_fit(
    session_data_names,
    unit_filter,
    input_unit_filter,
    window_config,
    data_format,
    trial_filters,
    train_test_option,
    random_state,
    n_continuous_states,
    ns_discrete_states,
    n_iters,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    inference_types=None):
    """
    Compute and plot per-condition forecast R² for LDS (d=1) vs rSLDS (d>1).

    Loads model pkl files (joint_SLDS_simple format: no fold prefix),
    runs forecast inference, and computes R² per condition per session.
    Produces a grouped bar chart showing R² vs number of discrete states.
    """
    import utils.utils_inference as utils_inference
    import utils.utils_decoding as utils_decoding

    if inference_types is None:
        inference_types = ['forecast']

    task_name = session_data_names[0].split('_')[-1]

    rows = []

    for session_data_name in session_data_names:
        print(f'\nSession: {session_data_name}')

        # Load data to get trial lengths and firing rates
        data_loader = utils_processing.DataLoaderDuo(
            data_dir, results_dir, session_data_name,
            unit_filter, input_unit_filter, window_config, trial_filters)
        data_loader.load_firing_rate_data()
        data_loader.load_cursor_data()
        from vis_config import session_target_radii
        data_loader.remove_target_overlap(
            target_radius=session_target_radii[session_data_name])

        (fr_fast, fr_slow,
         input_fr_fast, input_fr_slow,
         _, _,
         trial_lengths_fast, trial_lengths_slow,
         _, _, _, _) = data_loader.reformat_firing_rate_data(
            data_format, trial_length_filter_percentile=90)

        # Get model result directories
        fast_model_dir, slow_model_dir = data_loader.get_model_result_dirs(
            time_offset=None,
            data_format=data_format,
            train_test=train_test_option,
            model_type='rSLDS',
            dynamics_class=dynamics_class,
            emission_class=emission_class,
            init_type=init_type,
            subspace_type=subspace_type,
            alpha=alpha,
            check_existence=False)

        for n_discrete_states in ns_discrete_states:
            # Build model file name (joint_SLDS_simple format — no fold)
            model_save_name = '_'.join([
                f'r{random_state}',
                f's{n_continuous_states}',
                f'd{n_discrete_states}',
                f'i{n_iters}'])

            fast_path = os.path.join(fast_model_dir, model_save_name + '.pkl')
            slow_path = os.path.join(slow_model_dir, model_save_name + '.pkl')

            if not os.path.exists(fast_path) or not os.path.exists(slow_path):
                print(f'  [SKIP] d={n_discrete_states}: model files not found')
                print(f'    {fast_path}')
                continue

            with open(fast_path, 'rb') as f:
                res_fast = pickle.load(f)
            with open(slow_path, 'rb') as f:
                res_slow = pickle.load(f)

            model_fast = res_fast['model']
            model_slow = res_slow['model']

            # Use test continuous/discrete states from model results
            cont_fast = res_fast['test_continuous_states']
            cont_slow = res_slow['test_continuous_states']
            disc_fast = res_fast['test_discrete_states']
            disc_slow = res_slow['test_discrete_states']

            for inf_type in inference_types:
                if inf_type == 'forecast':
                    # Forecast: dynamics -> latent next -> emissions -> observed
                    pred_fast = utils_inference.forecast_inference_SLDS(
                        cont_fast, disc_fast, trial_lengths_fast,
                        model_fast.dynamics, model_fast.emissions, input_fr_fast)
                    pred_slow = utils_inference.forecast_inference_SLDS(
                        cont_slow, disc_slow, trial_lengths_slow,
                        model_slow.dynamics, model_slow.emissions, input_fr_slow)

                    # Predicted: t -> t+1, so compare to fr[1:]
                    pred_fast_cat = np.concatenate(
                        [p[:-1] for p in pred_fast], axis=0)
                    pred_slow_cat = np.concatenate(
                        [p[:-1] for p in pred_slow], axis=0)
                    true_fast_cat = np.concatenate(
                        [fr_fast[i][1:] for i in range(len(fr_fast))], axis=0)
                    true_slow_cat = np.concatenate(
                        [fr_slow[i][1:] for i in range(len(fr_slow))], axis=0)

                elif inf_type == 'emissions':
                    # Emissions: latent -> observed (no dynamics step)
                    # Compute manually to handle None inputs properly
                    Cs = model_fast.emissions.Cs[0]  # (N, D)
                    ds = model_fast.emissions.ds[0]  # (N,)
                    pred_fast = [x @ Cs.T + ds for x in cont_fast]
                    Cs = model_slow.emissions.Cs[0]
                    ds = model_slow.emissions.ds[0]
                    pred_slow = [x @ Cs.T + ds for x in cont_slow]

                    pred_fast_cat = np.concatenate(pred_fast, axis=0)
                    pred_slow_cat = np.concatenate(pred_slow, axis=0)
                    true_fast_cat = np.concatenate(
                        [fr_fast[i] for i in range(len(fr_fast))], axis=0)
                    true_slow_cat = np.concatenate(
                        [fr_slow[i] for i in range(len(fr_slow))], axis=0)

                else:
                    print(f'  Unknown inference type: {inf_type}')
                    continue

                r2_fast = utils_decoding.r2(true_fast_cat, pred_fast_cat)
                r2_slow = utils_decoding.r2(true_slow_cat, pred_slow_cat)

                model_label = 'LDS' if n_discrete_states == 1 else f'rSLDS (K={n_discrete_states})'
                print(f'  d={n_discrete_states} ({model_label}) {inf_type}: '
                      f'R²_{trial_filters[0]}={r2_fast:.4f}, '
                      f'R²_{trial_filters[1]}={r2_slow:.4f}')

                rows.append({
                    'session': session_data_name,
                    'unit_filter': unit_filter,
                    'n_discrete_states': n_discrete_states,
                    'model_label': model_label,
                    'inference_type': inf_type,
                    f'r2_{trial_filters[0]}': r2_fast,
                    f'r2_{trial_filters[1]}': r2_slow,
                })

    if not rows:
        print('No results computed.')
        return

    df = pd.DataFrame(rows)

    # Save CSV
    csv_name = f'{task_name}_lds_vs_rslds_r2_{unit_filter}.csv'
    csv_path = os.path.join(vis_dir, csv_name)
    df.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path}')

    # Plot: grouped bar chart per inference type
    for inf_type in inference_types:
        df_inf = df[df['inference_type'] == inf_type]
        if df_inf.empty:
            continue

        # Average over sessions
        df_avg = df_inf.groupby(['n_discrete_states', 'model_label']).agg(
            r2_cond1_mean=(f'r2_{trial_filters[0]}', 'mean'),
            r2_cond1_se=(f'r2_{trial_filters[0]}', 'sem'),
            r2_cond2_mean=(f'r2_{trial_filters[1]}', 'mean'),
            r2_cond2_se=(f'r2_{trial_filters[1]}', 'sem'),
        ).reset_index().sort_values('n_discrete_states')

        x = np.arange(len(df_avg))
        width = 0.35

        fig, ax = plt.subplots(figsize=(50 * mm, 35 * mm))

        bars1 = ax.bar(x - width/2, df_avg['r2_cond1_mean'], width,
                       yerr=df_avg['r2_cond1_se'],
                       label=trial_filters[0], color='#87CCE6',
                       edgecolor='none', capsize=2, error_kw={'linewidth': 0.5})
        bars2 = ax.bar(x + width/2, df_avg['r2_cond2_mean'], width,
                       yerr=df_avg['r2_cond2_se'],
                       label=trial_filters[1], color='#FDA058',
                       edgecolor='none', capsize=2, error_kw={'linewidth': 0.5})

        ax.set_xticks(x)
        ax.set_xticklabels(df_avg['model_label'], fontsize=5, rotation=30, ha='right')
        ax.set_ylabel(f'{inf_type} R²', fontsize=6)
        ax.tick_params(axis='both', labelsize=5)
        ax.legend(fontsize=5, frameon=False)
        sns.despine(ax=ax, top=True, right=True)

        fig.tight_layout()

        n_sess = len(session_data_names)
        fig_name = (f'{task_name}_{n_sess}sessions_lds_vs_rslds_'
                    f'{inf_type}_{unit_filter}.pdf')
        fig_path = os.path.join(vis_dir, fig_name)
        fig.savefig(fig_path, dpi=600, transparent=True, bbox_inches=None)
        plt.close(fig)
        print(f'Saved: {fig_path}')



def _test_against_baseline(values, baseline, method):
    """Two-sample test of values > baseline.  Returns raw p-value."""
    values = values[np.isfinite(values)]
    baseline = baseline[np.isfinite(baseline)]
    if method == 'wilcoxon':
        if len(values) < 3 or len(baseline) < 3:
            return 1.0
        _, p = mannwhitneyu(values, baseline, alternative='greater')
        return p
    elif method == 'ttest':
        if len(values) < 3 or len(baseline) < 3:
            return 1.0
        _, p = ttest_ind(values, baseline, alternative='greater')
        return p
    elif method == 'permutation':
        if len(values) < 5 or len(baseline) < 5:
            return 1.0
        observed = np.mean(values) - np.mean(baseline)
        pooled = np.concatenate([values, baseline])
        n_vals = len(values)
        n_perm = 5000
        rng = np.random.default_rng(42)
        count = 0
        for _ in range(n_perm):
            perm = rng.permutation(pooled)
            if np.mean(perm[:n_vals]) - np.mean(perm[n_vals:]) >= observed:
                count += 1
        return (count + 1) / (n_perm + 1)
    else:
        raise ValueError(f'Unknown significance method: {method}')


def cross_condition_significance_crossnobis(
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
    correction_method='fdr_bh',
    significance_method='wilcoxon',
    per_target=True,
    show_individual_targets=False,
    plot_heatmap=True,
    downsample_factor=1,
    supplement_format=False,
):
    """Test whether neural responses differ between conditions at each time.

    For each session (and optionally each target direction), computes the
    cross-validated Mahalanobis distance (crossnobis) between conditions
    using leave-one-out CV (each trial is its own fold).

    Significance is tested against the baseline (pre-movement) crossnobis
    level: for each observation, the mean crossnobis over the baseline
    period is subtracted, and the result is tested > 0.

    Both the diagonal (t vs t) and the full (t1 vs t2) matrix are tested.

    Parameters
    ----------
    significance_method : str
        'wilcoxon'    – one-sample Wilcoxon signed-rank (median > 0)
        'ttest'       – one-sample t-test (mean > 0)
        'permutation' – sign-flip permutation test (mean > 0)
    per_target : bool
        If True, stratify by target direction — one observation per
        (session, target).  If False, pool all trials within each session
        regardless of target — one observation per session.
    show_individual_targets : bool
        If True and per_target is True, show individual (session, target)
        lines colored by target using target_color_palette_8.
        If False, show per-session average lines in gray.
    plot_heatmap : bool
        If True, also produce a (t1, t2) significance heatmap.
    downsample_factor : int
        Temporal downsampling factor (default 1 = no downsampling).
        E.g. 10 keeps every 10th time bin, reducing computation ~100x.
    """

    task_name = session_data_names[0].split('_')[-1]
    post_reach_idx = int(post_reach_time_buffer / time_step)
    pre_start_idx = int(pre_start_time_buffer / time_step)

    # Apply downsampling to time parameters
    if downsample_factor > 1:
        effective_time_step = time_step * downsample_factor
        pre_start_idx = pre_start_idx // downsample_factor
        post_reach_idx = post_reach_idx // downsample_factor
        print(f'  Downsampling {downsample_factor}x: effective dt='
              f'{effective_time_step*1000:.1f}ms')
    else:
        effective_time_step = time_step

    # ---- Per-session: load from cache or compute crossnobis matrices ----------
    # Matrices are cached at full per-session size; global truncation happens
    # after all sessions are collected.
    all_matrices = []    # list of (T1_sess, T2_sess) arrays — one per observation
    obs_target_ids = []  # target id for each observation (None if pooled)
    crossnobis_diags_session = []

    pt_str = '_per_target' if per_target else ''
    ds_str = f'_ds{downsample_factor}' if downsample_factor > 1 else ''
    cache_key = f'stats_obs_stack{ds_str}'

    print(f'  {len(session_data_names)} sessions, CV=LOO, '
          f'method={significance_method}, per_target={per_target}')

    for i_sess, session_data_name in enumerate(session_data_names):
        cache_dir = os.path.join(
            results_dir, session_data_name,
            'crossnobis_matrices', unit_filter, window_config)
        cache_path = os.path.join(
            cache_dir,
            f'cross_{trial_filters[0]}_{trial_filters[1]}{pt_str}.pkl')

        # ── Try cache ──────────────────────────────────────────────────────
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as fh:
                cached_data = pickle.load(fh)
            if cache_key in cached_data:
                sess_mats, sess_tgt_ids = cached_data[cache_key]
                print(f'    session {i_sess}: loaded from cache '
                      f'({len(sess_mats)} observations)')
                all_matrices.extend(sess_mats)
                obs_target_ids.extend(sess_tgt_ids)
                if sess_mats:
                    crossnobis_diags_session.append(
                        np.mean([np.diag(m) for m in sess_mats], axis=0))
                continue

        # ── Cache miss: load firing rates and compute ───────────────────────
        fr_by_cond = {}
        tgt_by_cond = {}
        angle_by_cond = {}
        for trial_filter in trial_filters:
            dl = utils_processing.DataLoader(
                data_dir, results_dir, session_data_name,
                unit_filter, None, window_config, trial_filter)
            dl.load_firing_rate_data()
            fr, *_ = dl.reformat_firing_rate_data(
                data_format, index_buffer=post_reach_idx,
                trial_length_filter_percentile=truncate_percentile)
            fr_by_cond[trial_filter] = fr
            tgt_by_cond[trial_filter] = dl.get_target_indices()
            angle_by_cond[trial_filter] = dl.get_target_angles()

        fr_1 = fr_by_cond[trial_filters[0]]
        fr_2 = fr_by_cond[trial_filters[1]]
        tgt_1 = tgt_by_cond[trial_filters[0]]
        tgt_2 = tgt_by_cond[trial_filters[1]]
        ang_1 = angle_by_cond[trial_filters[0]]
        ang_2 = angle_by_cond[trial_filters[1]]

        if downsample_factor > 1:
            fr_1 = fr_1[:, ::downsample_factor, :]
            fr_2 = fr_2[:, ::downsample_factor, :]

        print(f'    session {i_sess}: n_{trial_filters[0]}={fr_1.shape[0]}, '
              f'n_{trial_filters[1]}={fr_2.shape[0]}, N={fr_1.shape[2]}')

        sess_mats = []
        sess_tgt_ids = []

        if per_target:
            # Try matching by target index first; if no overlap (e.g.
            # RadialGrid near/far have disjoint indices), match by angle.
            common_indices = np.intersect1d(np.unique(tgt_1), np.unique(tgt_2))
            if len(common_indices) > 0:
                match_key_1, match_key_2 = tgt_1, tgt_2
                unique_keys = common_indices
                key_to_color = {k: int(k) for k in unique_keys}
                print(f'    session {i_sess}: matching by target index')
            else:
                match_key_1, match_key_2 = ang_1, ang_2
                unique_keys = np.intersect1d(
                    np.unique(ang_1), np.unique(ang_2))
                key_to_color = {k: int(k) + 1 for k in unique_keys}
                print(f'    session {i_sess}: matching by target angle '
                      f'({len(unique_keys)} common angles)')

            for key in unique_keys:
                m1 = match_key_1 == key
                m2 = match_key_2 == key
                if m1.sum() < 2 or m2.sum() < 2:
                    continue
                _, _, _, cross_matrices = \
                    utils_processing.compute_cross_condition_crossnobis_matrix(
                        fr_1[m1], fr_2[m2])
                avg_mat = cross_matrices.mean(axis=0)
                sess_mats.append(avg_mat)
                sess_tgt_ids.append(key_to_color[key])

            print(f'    session {i_sess}: {len(sess_mats)} target matrices '
                  f'({len(unique_keys)} keys)')
        else:
            if fr_1.shape[0] < 2 or fr_2.shape[0] < 2:
                print(f'    session {i_sess}: skipped (too few trials)')
            else:
                _, _, _, cross_matrices = \
                    utils_processing.compute_cross_condition_crossnobis_matrix(
                        fr_1, fr_2)
                avg_mat = cross_matrices.mean(axis=0)
                sess_mats.append(avg_mat)
                sess_tgt_ids.append(None)
                print(f'    session {i_sess}: pooled all trials '
                      f'(n1={fr_1.shape[0]}, n2={fr_2.shape[0]})')

        # ── Save to cache ───────────────────────────────────────────────────
        os.makedirs(cache_dir, exist_ok=True)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as fh:
                cached_data = pickle.load(fh)
        else:
            cached_data = {}
        cached_data[cache_key] = (sess_mats, sess_tgt_ids)
        with open(cache_path, 'wb') as fh:
            pickle.dump(cached_data, fh)
        print(f'    session {i_sess}: saved to cache')

        all_matrices.extend(sess_mats)
        obs_target_ids.extend(sess_tgt_ids)
        if sess_mats:
            crossnobis_diags_session.append(
                np.mean([np.diag(m) for m in sess_mats], axis=0))

    n_obs = len(all_matrices)
    print(f'  {n_obs} total observations (session x target)')

    if n_obs == 0:
        print('  ERROR: no observations at all. Skipping.')
        return

    # Derive common time dimensions from the collected matrices
    min_T1 = min(m.shape[0] for m in all_matrices)
    min_T2 = min(m.shape[1] for m in all_matrices)
    min_T = min(min_T1, min_T2)
    print(f'  T1={min_T1}, T2={min_T2}')

    # Stack into (n_obs, T1, T2)
    obs_stack = np.array([m[:min_T1, :min_T2] for m in all_matrices])
    # Diagonal: extract min(T1,T2) elements from each observation
    diag_matrix = np.array([np.diag(m[:min_T, :min_T]) for m in obs_stack])  # (n_obs, min_T)
    crossnobis_avg_diag = diag_matrix.mean(axis=0)
    crossnobis_avg_full = obs_stack.mean(axis=0)  # (T1, T2)

    # ---- Diagonal significance at multiple alpha levels -----------------------
    # Two-sample test: at each time bin, compare the n_obs crossnobis values
    # against the n_obs baseline values (per-observation mean over pre-movement).
    alphas = [0.05, 0.01, 0.001]
    times = (np.arange(min_T) - pre_start_idx) * effective_time_step

    # Baseline: per-observation mean over pre-movement bins → (n_obs,)
    baseline_diag = diag_matrix[:, :pre_start_idx].mean(axis=1)
    print(f'  Baseline crossnobis (mean over {pre_start_idx} bins): '
          f'mean={baseline_diag.mean():.4f}, std={baseline_diag.std():.4f}')

    pvals_diag = np.ones(min_T)
    for t in range(min_T):
        pvals_diag[t] = _test_against_baseline(
            diag_matrix[:, t], baseline_diag, significance_method)

    # FDR correction
    _, pvals_diag_corrected, _, _ = multipletests(
        pvals_diag, alpha=alphas[0], method=correction_method)

    # ---- Diagonal line plot with multi-level significance bars ----------------
    fig, ax = plt.subplots(figsize=((35 if supplement_format else 90) * mm, 45 * mm))
    ax.plot(times, crossnobis_avg_diag, color='k', lw=0.8)
    ax.axhline(0, color='gray', linestyle=':', lw=0.5, alpha=0.5)

    if show_individual_targets and per_target:
        # Average across sessions per target direction → one line per target
        unique_tgt_ids = sorted(set(obs_target_ids) - {None})
        for tgt_id in unique_tgt_ids:
            idxs = [i for i, t in enumerate(obs_target_ids) if t == tgt_id]
            avg_diag = np.mean(
                [np.diag(obs_stack[i, :min_T, :min_T]) for i in idxs], axis=0)
            color = target_color_palette_8[tgt_id - 1]
            ax.plot(times, avg_diag, color=color, lw=0.5, alpha=0.3)
    else:
        for diag in crossnobis_diags_session:
            ax.plot(times, diag, color='gray', lw=0.3, alpha=0.2)

    # Multi-level significance bars
    y0, y1 = ax.get_ylim()
    bar_gap = 0.07 * (y1 - y0)
    bar_colors = {0.05: '#888888', 0.01: '#444444', 0.001: '#000000'}
    bar_labels = {0.05: 'p<0.05', 0.01: 'p<0.01', 0.001: 'p<0.001'}

    for i_alpha, alpha in enumerate(alphas):
        reject_alpha = pvals_diag_corrected < alpha
        bar_y = y1 + (i_alpha + 1) * bar_gap
        for k, g in itertools.groupby(enumerate(reject_alpha), key=lambda x: x[1]):
            if k:
                idxs = [idx for idx, _ in g]
                ax.hlines(bar_y, times[idxs[0]], times[idxs[-1]],
                          color=bar_colors[alpha], linewidth=1.0, alpha=0.8)
        # Label on the right
        ax.text(times[-1] + 0.02 * (times[-1] - times[0]), bar_y,
                bar_labels[alpha], fontsize=5, va='center',
                color=bar_colors[alpha])

    ax.set_ylim(y0, y1 + (len(alphas) + 1) * bar_gap)

    # Trial start line at 0s (red dashed)
    ax.axvline(0, color='red', linestyle='--', lw=0.5, alpha=0.5)

    if visual_delay_time is not None:
        ax.axvline(visual_delay_time, color='black', linestyle='--',
                   lw=0.5, alpha=0.5)
    if peak_time is not None:
        ax.axvline(peak_time, color='black', linestyle=':',
                   lw=0.5, alpha=0.5)

    ax.set_xlabel('Time (s)', fontsize=7)
    ax.set_ylabel(r'Crossnobis distance ($\Delta$Hz$^2$)', fontsize=7)
    ax.tick_params(axis='both', labelsize=5)
    sns.despine(ax=ax, top=True, right=True)
    fig.tight_layout()

    subject_task = (session_data_names[0].split('_')[0][-2:]
                    + '_' + task_name)
    tgt_str = '_perTarget' if per_target else '_pooled'
    fig_name = (f'{subject_task}_cross_condition_significance'
                f'_{significance_method}{tgt_str}'
                f'_{unit_filter}_{window_config}.pdf')
    fig_path = os.path.join(vis_dir, fig_name)
    fig.savefig(fig_path, dpi=600, transparent=True, bbox_inches=None)
    plt.close(fig)
    print(f'Saved: {fig_path}')

    for alpha in alphas:
        n_sig = (pvals_diag_corrected < alpha).sum()
        pct = 100 * n_sig / min_T
        print(f'  alpha={alpha}: {n_sig}/{min_T} ({pct:.1f}%) significant')

    # Save diagonal CSV
    df = pd.DataFrame({
        'time': times,
        'crossnobis_avg': crossnobis_avg_diag,
        'pval_raw': pvals_diag,
        'pval_corrected': pvals_diag_corrected,
    })
    csv_name = (f'{subject_task}_cross_condition_significance'
                f'_{significance_method}{tgt_str}'
                f'_{unit_filter}_{window_config}.csv')
    csv_path = os.path.join(vis_dir, csv_name)
    df.to_csv(csv_path, index=False)
    print(f'Saved: {csv_path}')

    # ---- Full (t1, t2) significance heatmap -----------------------------------
    if plot_heatmap:
        print('  Computing full (t1, t2) significance heatmap...')
        # Rectangular: rows=cond1 (T1), cols=cond2 (T2)
        # Baseline: per-observation mean over the pre-movement block → (n_obs,)
        baseline_full = obs_stack[:, :pre_start_idx, :pre_start_idx].mean(
            axis=(1, 2))

        pvals_full = np.ones((min_T1, min_T2))
        for t1 in range(min_T1):
            for t2 in range(min_T2):
                p = _test_against_baseline(
                    obs_stack[:, t1, t2], baseline_full, significance_method)
                pvals_full[t1, t2] = p
            if (t1 + 1) % 200 == 0:
                print(f'    row {t1 + 1}/{min_T1}')

        # FDR correction over all entries
        pvals_flat = pvals_full.ravel()
        _, pvals_flat_corrected, _, _ = multipletests(
            pvals_flat, alpha=alphas[0], method=correction_method)
        pvals_full_corrected = pvals_flat_corrected.reshape(min_T1, min_T2)

        # Plot: p-value heatmap — white where p >= 0.05, brighter where smaller
        from matplotlib.colors import ListedColormap, BoundaryNorm

        bounds = [0, 0.0001, 0.001, 0.01, 0.05, 1.0]
        # white = non-significant (p >= 0.05)
        # increasingly vivid for smaller p-values
        colors_list = ['#d62728', '#ff7f0e', '#ffdd57', '#fff3b0', '#ffffff']
        cmap_sig = ListedColormap(colors_list)
        norm = BoundaryNorm(bounds, len(colors_list))

        fig, ax = plt.subplots(1, 1, figsize=((45 if supplement_format else 90) * mm, 45 * mm))

        times_1 = (np.arange(min_T1) - pre_start_idx) * effective_time_step
        times_2 = (np.arange(min_T2) - pre_start_idx) * effective_time_step
        im = ax.pcolormesh(
            times_2, times_1, pvals_full_corrected,
            cmap=cmap_sig, norm=norm, rasterized=True)

        cbar = fig.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_ticks([0.0001, 0.001, 0.01, 0.05])
        cbar.set_ticklabels(['0.0001', '0.001', '0.01', '0.05'])
        cbar.set_label(r'$p_{\mathrm{corr}}$', fontsize=5)
        cbar.ax.tick_params(labelsize=5)

        if task_name == 'RadialGrid':
            ax.set_ylabel(f'{trial_filters[0].capitalize()} time (s)', fontsize=5)
            ax.set_xlabel(f'{trial_filters[1].capitalize()} time (s)', fontsize=5)
        else:
            label_map = {'fast': 'Ballistic', 'slow': 'Sustained'}
            ax.set_ylabel(f'{label_map.get(trial_filters[0], trial_filters[0])} time (s)', fontsize=5)
            ax.set_xlabel(f'{label_map.get(trial_filters[1], trial_filters[1])} time (s)', fontsize=5)

        ax.tick_params(axis='both', labelsize=5)
        ax.invert_yaxis()

        # Visual delay / peak markers
        if pre_start_time_buffer > 0:
            vis_time = visual_delay_time
            ax.axhline(y=vis_time, color='gray', linestyle='--',
                       lw=size_line_thin, alpha=0.7)
            ax.axvline(x=vis_time, color='gray', linestyle='--',
                       lw=size_line_thin, alpha=0.7)
            ax.axhline(y=peak_time, color='gray', linestyle=':',
                       lw=size_line_thin, alpha=0.7)
            ax.axvline(x=peak_time, color='gray', linestyle=':',
                       lw=size_line_thin, alpha=0.7)

        fig.tight_layout()
        hm_name = (f'{subject_task}_cross_condition_significance_heatmap'
                   f'_{significance_method}{tgt_str}'
                   f'_{unit_filter}_{window_config}.pdf')
        hm_path = os.path.join(vis_dir, hm_name)
        fig.savefig(hm_path, dpi=600, transparent=True, bbox_inches=None)
        plt.close(fig)
        print(f'Saved: {hm_path}')
