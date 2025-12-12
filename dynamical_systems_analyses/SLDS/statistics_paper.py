"""Statistical analyses and helper routines used in the dynamics paper figures."""

import os
import ipdb
import pickle
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon, mannwhitneyu, ttest_1samp, ttest_rel, f_oneway, ttest_ind
from statsmodels.stats.multitest import multipletests

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cross_decomposition import PLSRegression

import dynamical_systems_analyses.utils.utils_processing as utils_processing
import dynamical_systems_analyses.utils.utils_decoding as utils_decoding
import dynamical_systems_analyses.utils.utils_vis as utils_vis
import dynamical_systems_analyses.SLDS.config as config
from vis_config import *



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



def time_crossnobis_RDM_superdiagonal_statistics(
    session_data_names: str,
    unit_filters: str,
    window_config: str,
    time_step: float,
    trial_filters: list[str],
    truncate_percentile: int = 90,
    pre_start_time_buffer: float = 0.2,
    post_reach_time_buffer: float = 0.5,
    superdiagonal_order: int = 1):
    

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

        print(f'Peak vs Baseline:  Mean Δ={mean_diff:.3f} ± {half_w:.3f} '
              f'(95 % CI {ci_peak[0]:.3f}–{ci_peak[1]:.3f}), '
              f"Cohen's dₚ = {d_peak:.3f}")

        # ─── Effect size + 95 % CI for Return vs Baseline ────────────────────
        diff_ret  = return_trial - baseline_trial
        n         = diff_ret.size
        mean_diff = diff_ret.mean()
        sd_diff   = diff_ret.std(ddof=1)
        d_ret     = mean_diff / sd_diff                      # Cohen's d_z

        t_crit    = t.ppf(0.975, df=n-1)
        half_w    = t_crit * sd_diff / np.sqrt(n)
        ci_ret    = (mean_diff - half_w, mean_diff + half_w)

        print(f'Return vs Baseline: Mean Δ={mean_diff:.3f} ± {half_w:.3f} '
              f'(95 % CI {ci_ret[0]:.3f}–{ci_ret[1]:.3f}), '
              f"Cohen's dₚ = {d_ret:.3f}")
        
    
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

        ipdb.set_trace()

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

        print(f'Peak vs Baseline:  Mean Δ={mean_diff:.3f} ± {half_w:.3f} '
              f'(95 % CI {ci_peak[0]:.3f}–{ci_peak[1]:.3f}), '
              f"Cohen's dₚ = {d_peak:.3f}")

        # ─── Effect size + 95 % CI for Return vs Baseline ────────────────────
        diff_ret  = return_trial - baseline_trial
        n         = diff_ret.size
        mean_diff = diff_ret.mean()
        sd_diff   = diff_ret.std(ddof=1)
        d_ret     = mean_diff / sd_diff                      # Cohen's d_z

        t_crit    = t.ppf(0.975, df=n-1)
        half_w    = t_crit * sd_diff / np.sqrt(n)
        ci_ret    = (mean_diff - half_w, mean_diff + half_w)

        print(f'Return vs Baseline: Mean Δ={mean_diff:.3f} ± {half_w:.3f} '
              f'(95 % CI {ci_ret[0]:.3f}–{ci_ret[1]:.3f}), '
              f"Cohen's dₚ = {d_ret:.3f}")


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

        # ─── effect size ──────────────────────────────────────────────────
        pooled_sd = np.sqrt(((steady_phase_durations_fast.std(ddof=1)**2 + steady_phase_durations_slow.std(ddof=1)**2) / 2))
        d = (np.mean(steady_phase_durations_fast) - np.mean(steady_phase_durations_slow)) / pooled_sd
        print(f"Cohen d = {d:.3f}")


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

                # Load results if they exist
                crossnobis_save_path = os.path.join(
                    results_dir,
                    session_data_name,
                    'crossnobis_matrices',
                    unit_filter,
                    window_config,
                    trial_filter + '.pkl'
                )
                print(f'Processing {session_data_name}, {unit_filter}, {trial_filter}')
                if os.path.exists(crossnobis_save_path):
                    with open(crossnobis_save_path, 'rb') as f:
                        data = pickle.load(f)
                    crossnobis_matrices_truncate_end = data['crossnobis_matrices_truncate_end']
                    crossnobis_matrices_truncate_front = data['crossnobis_matrices_truncate_front']
                    full_trial_lengths = data['full_trial_lengths']

                    crossnobis_matrices_all_truncate_end[i_tf].append(crossnobis_matrices_truncate_end)
                    crossnobis_matrices_all_truncate_front[i_tf].append(crossnobis_matrices_truncate_front)
                    full_trial_lengths_all[i_tf].append(full_trial_lengths)

                    min_lengths_truncate_end[i_tf] = min(min_lengths_truncate_end[i_tf], crossnobis_matrices_truncate_end.shape[1])
                    min_lengths_truncate_front[i_tf] = min(min_lengths_truncate_front[i_tf], crossnobis_matrices_truncate_front.shape[1])

                    continue

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

                # Crossnobis -------------------------------------------------------
                crossnobis_matrix_truncate_end, crossnobis_matrices_truncate_end = utils_processing.compute_crossnobis_matrix(
                    fr_simple_truncate_end, time_step=time_step)

                crossnobis_matrix_truncate_front, crossnobis_matrices_truncate_front = utils_processing.compute_crossnobis_matrix(
                    fr_simple_truncate_front, time_step=time_step)

                crossnobis_matrices_all_truncate_end[i_tf].append(crossnobis_matrices_truncate_end)
                crossnobis_matrices_all_truncate_front[i_tf].append(crossnobis_matrices_truncate_front)
                full_trial_lengths_all[i_tf].append(full_trial_lengths)

                min_lengths_truncate_end[i_tf] = min(min_lengths_truncate_end[i_tf], crossnobis_matrix_truncate_end.shape[0])
                min_lengths_truncate_front[i_tf] = min(min_lengths_truncate_front[i_tf], crossnobis_matrix_truncate_front.shape[0])

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


    # analyze_neural_speed_phase(
    #     neural_speed_all_truncate_end,
    #     full_trial_lengths_all)









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
