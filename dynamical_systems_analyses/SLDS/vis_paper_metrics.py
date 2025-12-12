import os
import ipdb
import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import fdrcorrection

import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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



alpha_point = 0.8
size_point  = 100
alpha_line_thin  = 0.3
alpha_line       = 0.6
alpha_line_thick = 0.9
size_line_thin  = 1
size_line       = 3
size_line_thick = 5
label_fontsize = 18



theme_coral_light = (255/255, 204/255, 204/255)
theme_coral_mid   = (247/255, 156/255, 156/255)
theme_coral_dark  = (232/255, 107/255, 107/255)
theme_green_light = (206/255, 242/255, 238/255) 
theme_green_mid   = (120/255, 204/255, 194/255)
theme_green_dark  = (74/255,  189/255, 175/255)
theme_blue_light = sns.color_palette('Blues')[0]
theme_blue_mid   = sns.color_palette('Blues')[2]
theme_blue_dark  = sns.color_palette('Blues')[3]
theme_orange_light = sns.color_palette('Oranges')[0]
theme_orange_mid   = sns.color_palette('Oranges')[2]
theme_orange_dark  = sns.color_palette('Oranges')[3]

corals  = [theme_coral_light, theme_coral_mid, theme_coral_dark]
greens  = [theme_green_light, theme_green_mid, theme_green_dark]
blues   = [theme_blue_light, theme_blue_mid, theme_blue_dark]
oranges = [theme_orange_light, theme_orange_mid, theme_orange_dark]


trial_filter_colors_palettes = {
    'fast': blues,
    'slow': oranges,
    'far':  corals,
    'near': greens,
}

trial_filter_colors = {
    'fast': blues[1],
    'slow': oranges[1],
    'far':  corals[1],
    'near': greens[1],
}

discrete_state_cmap = sns.color_palette('spring', as_cmap=True)

trial_filter_cmaps = {
    # 'slow': sns.color_palette('Oranges', as_cmap=True),
    # 'fast': sns.color_palette('Blues', as_cmap=True),

    # 'slow': mcolors.LinearSegmentedColormap.from_list('trial_filter_cmap_slow', [sns.color_palette('Oranges')[0], sns.color_palette('Oranges')[3]]),
    # 'fast': mcolors.LinearSegmentedColormap.from_list('trial_filter_cmap_fast', [sns.color_palette('Blues')[3],   sns.color_palette('Blues')[0]]),

    # 'slow': mcolors.LinearSegmentedColormap.from_list('trial_filter_cmap_slow', [sns.color_palette('Oranges')[3], sns.color_palette('Oranges')[0]]),
    # 'fast': mcolors.LinearSegmentedColormap.from_list('trial_filter_cmap_fast', [sns.color_palette('Blues')[3],   sns.color_palette('Blues')[0]]),
    'slow': mcolors.LinearSegmentedColormap.from_list('gray_r_cmap', [sns.color_palette('Greys')[3], sns.color_palette('Greys')[0]]),
    'fast': mcolors.LinearSegmentedColormap.from_list('gray_r_cmap', [sns.color_palette('Greys')[3], sns.color_palette('Greys')[0]]),
    'near': mcolors.LinearSegmentedColormap.from_list('trial_filter_cmap_near', [greens[2], greens[0]]),
    'far':  mcolors.LinearSegmentedColormap.from_list('trial_filter_cmap_far',  [corals[2], corals[0]]),
    # 'fast': mcolors.LinearSegmentedColormap.from_list('gray_cmap', [sns.color_palette('Greys')[0], sns.color_palette('Greys')[3]]),
    'gray_r_cmap': mcolors.LinearSegmentedColormap.from_list('gray_r_cmap', [sns.color_palette('Greys')[3], sns.color_palette('Greys')[0]]),
    'gray_cmap': mcolors.LinearSegmentedColormap.from_list('gray_cmap', [sns.color_palette('Greys')[0], sns.color_palette('Greys')[3]]),
}

trial_filter_colors_by_discrete_state = {
    # 'slow': [sns.color_palette('Oranges')[0], sns.color_palette('Oranges')[3]],
    # 'fast': [sns.color_palette('Blues')[0], sns.color_palette('Blues')[3]],

    'slow': [
        sns.color_palette('Oranges')[3], 
        sns.color_palette('Oranges')[0],
        sns.color_palette('Oranges')[1],
        sns.color_palette('Oranges')[2],
        sns.color_palette('Oranges')[4],
        sns.color_palette('Oranges')[5],
        sns.color_palette('Reds')[3],
        sns.color_palette('Reds')[0],
        sns.color_palette('Reds')[1],
        sns.color_palette('Reds')[2],
        sns.color_palette('Reds')[4],
        sns.color_palette('Reds')[5],
        sns.color_palette('Greys')[0],
        sns.color_palette('Greys')[1],
        sns.color_palette('Greys')[2],
        sns.color_palette('Greys')[3]],
    'fast': [
        sns.color_palette('Blues')[3], 
        sns.color_palette('Blues')[0],
        sns.color_palette('Blues')[1],
        sns.color_palette('Blues')[2],
        sns.color_palette('Blues')[4],
        sns.color_palette('Blues')[5],
        sns.color_palette('Greens')[3],
        sns.color_palette('Greens')[0],
        sns.color_palette('Greens')[1],
        sns.color_palette('Greens')[2],
        sns.color_palette('Greens')[4],
        sns.color_palette('Greens')[5],
        sns.color_palette('Purples')[0],
        sns.color_palette('Purples')[1],
        sns.color_palette('Purples')[2],
        sns.color_palette('Purples')[3]],
    'near': [greens[2], greens[0], greens[1]],
    'far': [corals[2], corals[0], corals[1]],
}

metric_plot_colors = {
    'rSLDS slow': '#C93735',
    'rSLDS fast': '#E59693',
    'LDS slow':   '#333A8C',
    'LDS fast':   '#989CC8',
    'PCA slow':   '#F8AB61',
    'PCA fast':   '#FCD7AF',
}

# discrete_state_colors = [
#     '#FD8D61',
#     '#5D63AE',
# ]

# discrete_state_colors = [
#     '#5D63AE',
#     '#FD8D61',
# ]

discrete_state_colors = sns.color_palette('Set2')
discrete_state_colors_with_PCA = [mcolors.to_rgb('skyblue')] + discrete_state_colors
continuous_state_colors = sns.color_palette('rocket', n_colors=n_ns_continuous_states)

trial_filter_name_conversion = {
    'fast': 'Ballistic',
    'slow': 'Sustained',
    'near': 'Near',
    'far':  'Far',
}

time_gradient_cmap = sns.color_palette('winter_r', as_cmap=True)
# time_gradient_cmap = sns.color_palette('hsv', as_cmap=True)
# time_gradient_cmap = sns.color_palette('summer', as_cmap=True)

caltech_orange = '#FF6C0C'

target_color_palette_8  = sns.color_palette('hls', 8)

discrete_state_quiver_cmaps = [
    sns.color_palette('crest_r',   as_cmap=True),
    sns.color_palette('flare_r',   as_cmap=True),
    sns.color_palette('rocket',    as_cmap=True), 
    sns.color_palette('mako',      as_cmap=True),
    sns.color_palette('viridis',   as_cmap=True),
    sns.color_palette('plasma',    as_cmap=True),
    sns.color_palette('inferno',   as_cmap=True),
    sns.color_palette('magma',     as_cmap=True),
    sns.color_palette('cividis',   as_cmap=True),
    sns.color_palette('cubehelix', as_cmap=True),
    sns.color_palette('Spectral',  as_cmap=True),
    sns.color_palette('coolwarm',  as_cmap=True),
    sns.color_palette('bwr',       as_cmap=True),
    sns.color_palette('GnBu',      as_cmap=True),
    sns.color_palette('BuPu',      as_cmap=True),
    sns.color_palette('YlGnBu',    as_cmap=True),
]

elbos_cmaps = {
    'rSLDS': sns.color_palette('rocket', len(ns_states)),
    'LDS':   sns.color_palette('mako',   len(ns_states)),
}



def plot_decoding_results_avg_session(
    unit_filter,
    input_unit_filter,
    data_format,
    label_format,
    train_test_option,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    train_or_test='test',
    form='heatmap'):

    print('Plotting decoding results...')

    task_name = session_data_names[0].split('_')[-1]

    xticks = np.arange(len(ns_states))
    xticklabels = ns_states

    rSLDS_decoding_name = '_'.join(map(str, [x for x in [
        'decoding',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        label_format,
        trial_filters,
        train_test_option,
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))
    
    PCA_decoding_name = '_'.join(map(str, [x for x in [
        'decoding',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        label_format,
        trial_filters,
        train_test_option,
        random_states,
        n_folds,
        ns_states,
        'PCA'] if x is not None]))
    

    rSLDS_decoding_slow_all = []
    rSLDS_decoding_fast_all = []
    PCA_decoding_slow_all   = []
    PCA_decoding_fast_all   = []

    for session_data_name in session_data_names:

        session_results_dir = os.path.join(results_dir, session_data_name)

        rSLDS_decoding = np.load(os.path.join(session_results_dir, rSLDS_decoding_name + '.npz'))
        rSLDS_decoding_slow = rSLDS_decoding['decoding_errors_slow']
        rSLDS_decoding_fast = rSLDS_decoding['decoding_errors_fast']
        # PCA_decoding = np.load(os.path.join(session_results_dir, PCA_decoding_name + '.npz'))
        # PCA_decoding_slow = PCA_decoding['decoding_errors_slow']
        # PCA_decoding_fast = PCA_decoding['decoding_errors_fast']

        ## Take the last n_iters, then add to all sessions
        rSLDS_decoding_slow = rSLDS_decoding_slow[..., -1]
        rSLDS_decoding_fast = rSLDS_decoding_fast[..., -1]
        # PCA_decoding_slow = PCA_decoding_slow
        # PCA_decoding_fast = PCA_decoding_fast

        rSLDS_decoding_slow_all.append(rSLDS_decoding_slow)
        rSLDS_decoding_fast_all.append(rSLDS_decoding_fast)
        # PCA_decoding_slow_all.append(PCA_decoding_slow)
        # PCA_decoding_fast_all.append(PCA_decoding_fast)

    ## Stack all sessions
    rSLDS_decoding_slow_all = np.stack(rSLDS_decoding_slow_all, axis=0)
    rSLDS_decoding_fast_all = np.stack(rSLDS_decoding_fast_all, axis=0)
    # PCA_decoding_slow_all = np.stack(PCA_decoding_slow_all, axis=0)
    # PCA_decoding_fast_all = np.stack(PCA_decoding_fast_all, axis=0)

    ## Compute mean and standard error over sessions, random states, and folds
    rSLDS_decoding_slow_all_mean = np.mean(rSLDS_decoding_slow_all, axis=(0, 2, 3))
    rSLDS_decoding_fast_all_mean = np.mean(rSLDS_decoding_fast_all, axis=(0, 2, 3))
    # PCA_decoding_slow_all_mean = np.mean(PCA_decoding_slow_all, axis=(0, 2, 3))
    # PCA_decoding_fast_all_mean = np.mean(PCA_decoding_fast_all, axis=(0, 2, 3))

    correction_factor = np.sqrt(len(session_data_names) * len(random_states) * n_folds)
    rSLDS_decoding_slow_all_se = np.std(rSLDS_decoding_slow_all, axis=(0, 2, 3)) / correction_factor
    rSLDS_decoding_fast_all_se = np.std(rSLDS_decoding_fast_all, axis=(0, 2, 3)) / correction_factor
    # PCA_decoding_slow_all_se = np.std(PCA_decoding_slow_all, axis=(0, 2, 3)) / correction_factor
    # PCA_decoding_fast_all_se = np.std(PCA_decoding_fast_all, axis=(0, 2, 3)) / correction_factor

    ## Concatenate all decoding results
    # decoding_slow_mean = np.concatenate([PCA_decoding_slow_all_mean[:, :, None], rSLDS_decoding_slow_all_mean], axis=-1)
    # decoding_fast_mean = np.concatenate([PCA_decoding_fast_all_mean[:, :, None], rSLDS_decoding_fast_all_mean], axis=-1)
    # decoding_slow_se = np.concatenate([PCA_decoding_slow_all_se[:, :, None], rSLDS_decoding_slow_all_se], axis=-1)
    # decoding_fast_se = np.concatenate([PCA_decoding_fast_all_se[:, :, None], rSLDS_decoding_fast_all_se], axis=-1)

    decoding_slow_mean = rSLDS_decoding_slow_all_mean
    decoding_fast_mean = rSLDS_decoding_fast_all_mean
    decoding_slow_se = rSLDS_decoding_slow_all_se
    decoding_fast_se = rSLDS_decoding_fast_all_se

    decoding_results = {
        'slow':
        # 'far':
            {
                'rSLDS_mean': rSLDS_decoding_slow_all_mean,
                # 'PCA_mean': PCA_decoding_slow_all_mean,
                'all_mean': decoding_slow_mean,
                'rSLDS_se': rSLDS_decoding_slow_all_se,
                # 'PCA_se': PCA_decoding_slow_all_se,
                'all_se': decoding_slow_se,
            },
        'fast':
        # 'near':
            {
                'rSLDS_mean': rSLDS_decoding_fast_all_mean,
                # 'PCA_mean': PCA_decoding_fast_all_mean,
                'all_mean': decoding_fast_mean,
                'rSLDS_se': rSLDS_decoding_fast_all_se,
                # 'PCA_se': PCA_decoding_fast_all_se,
                'all_se': decoding_fast_se,
            }
    }
    
    vmin = 0
    vmax = 90

    if train_or_test == 'train':
        train_or_test_id = 0
    elif train_or_test == 'test':
        train_or_test_id = 1
    
    ## Plot results
    if form == 'scatter':

        ## Use PCA results as baseline
        # fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # for i_tf, trial_filter in enumerate(trial_filters):

        #     decoding_results_rSLDS = decoding_results[trial_filter]['rSLDS'][train_or_test_id, ...]
        #     decoding_results_PCA   = decoding_results[trial_filter]['PCA'][train_or_test_id, ...]
        #     decoding_results_ = decoding_results_rSLDS - decoding_results_PCA[:, None]

        #     for i_ds in range(len(ns_discrete_states)):
                
        #         axs[i_tf].axhline(
        #             y=0,
        #             color='black',
        #             lw=size_line_thin,
        #             alpha=alpha_line_thin,
        #             linestyle='--')

        #         axs[i_tf].scatter(
        #             ns_states, 
        #             decoding_results_[:, i_ds], 
        #             color=discrete_state_colors[i_ds],
        #             alpha=alpha_line,
        #             s=size_point,
        #             )
                
        #         axs[i_tf].set_xticks(ns_states)

        ## Not using PCA results as baseline
        fig, axs = plt.subplots(1, 2, figsize=(45*mm, 22.5*mm), sharey=True)

        for i_tf, trial_filter in enumerate(trial_filters):

            decoding_results_ = decoding_results[trial_filter]['all_mean'][train_or_test_id, ...]

            ## Find the simplest model with performance better than best result + its standard error
            performance_reference = utils_vis.model_selection(
                decoding_results_, 
                decoding_results[trial_filter]['all_se'][train_or_test_id, ...],
                ns_states, 
                ns_discrete_states,
                includes_pca=True,
                higher_is_better=False)

            # axs[i_tf].axhline(
            #     y=performance_reference,
            #     color='black',
            #     lw=size_line_thin,
            #     alpha=alpha_line_thin,
            #     linestyle='--')

            for i_ds in reversed(range(len(ns_discrete_states))):

                ## Introduce jitter
                x = ns_states + np.random.uniform(-0.1, 0.1, size=len(ns_states))
                y = decoding_results_[:, i_ds]

                axs[i_tf].scatter(
                    x, 
                    y, 
                    color=discrete_state_colors[i_ds],
                    alpha=alpha_line,
                    s=2,
                    edgecolors='none',
                )

            axs[i_tf].set_xticks(ns_states)
            tick_labels = [str(x) if i % 2 == 0 else '' for i, x in enumerate(ns_states)]

            ## Set tick labels
            axs[i_tf].set_xticklabels(tick_labels)
            # axs[i_tf].set_xticklabels([])
            # axs[i_tf].set_yticklabels([])

            axs[i_tf].set_ylim(vmin, vmax)
            # axs[i_tf].set_xlabel('Number of continuous states', fontsize=5)
            # axs[i_tf].set_ylabel('Absolute angle error (degrees)', fontsize=5)

            ## Remove top and right spines
            axs[i_tf].spines['top'].set_visible(False)
            axs[i_tf].spines['right'].set_visible(False)


        # fig.subplots_adjust(bottom=0.2)

        # Define your custom markers for the legend
        # custom_markers = [
        #     Line2D([0], [0], color=discrete_state_colors_with_PCA[0], marker='o', linestyle='None', markeredgecolor='none', markersize=10, alpha=alpha_line),
        #     Line2D([0], [0], color=discrete_state_colors_with_PCA[1], marker='o', linestyle='None', markeredgecolor='none', markersize=10, alpha=alpha_line),
        #     Line2D([0], [0], color=discrete_state_colors_with_PCA[2], marker='o', linestyle='None', markeredgecolor='none', markersize=10, alpha=alpha_line),
        #     Line2D([0], [0], color=discrete_state_colors_with_PCA[3], marker='o', linestyle='None', markeredgecolor='none', markersize=10, alpha=alpha_line),
        #     Line2D([0], [0], color=discrete_state_colors_with_PCA[4], marker='o', linestyle='None', markeredgecolor='none', markersize=10, alpha=alpha_line),
        #     Line2D([0], [0], color=discrete_state_colors_with_PCA[5], marker='o', linestyle='None', markeredgecolor='none', markersize=10, alpha=alpha_line),
        #     Line2D([0], [0], color=discrete_state_colors_with_PCA[6], marker='o', linestyle='None', markeredgecolor='none', markersize=10, alpha=alpha_line),
        #     Line2D([0], [0], color=discrete_state_colors_with_PCA[7], marker='o', linestyle='None', markeredgecolor='none', markersize=10, alpha=alpha_line),
        #     Line2D([0], [0], color=discrete_state_colors_with_PCA[8], marker='o', linestyle='None', markeredgecolor='none', markersize=10, alpha=alpha_line)
        # ]

        # # Your legend labels remain the same
        # legend_labels = ['PCA', '1', '2', '3', '4', '6', '8', '12', '16']

        # # Create the legend with the custom markers
        # plt.legend(
        #     custom_markers, 
        #     legend_labels, 
        #     title='# of Discrete States',
        #     loc='upper center', 
        #     bbox_to_anchor=(-0.2, -0.15), 
        #     fancybox=False, 
        #     shadow=False, 
        #     ncol=9
        # )


    elif form == 'swarm':
    
        fig, axs = plt.subplots(1, 2, figsize=(30, 6))
    
        for i_tf, trial_filter in enumerate(trial_filters):

            decoding_results_ = decoding_results[trial_filter]['all'][train_or_test_id, ...]
    
            # Prepare data for seaborn
            data_list = []
            for i_ds in range(len(ns_discrete_states) + 1):
                for ns_state, result in zip(ns_states, decoding_results_[:, i_ds]):
                    data_list.append({
                        'ns_state': ns_state,
                        'decoding_result': result,
                        'discrete_state': i_ds
                    })
                    
            data = pd.DataFrame(data_list)
            
            # Plot using seaborn
            sns.swarmplot(
                x='ns_state', 
                y='decoding_result', 
                hue='discrete_state', 
                data=data, 
                ax=axs[i_tf],
                palette=discrete_state_colors_with_PCA
            )

            axs[i_tf].set_ylim(10, 30)


    fig.tight_layout()

    ## Write image
    session_data_names_str = str(len(session_data_names)) + '_sessions'

    img_name = '_'.join(map(str, [x for x in [
        task_name,
        session_data_names_str,
        'decoding',
        train_or_test,
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        label_format,
        trial_filters,
        train_test_option,
        dynamics_class,
        emission_class,
        init_type,
        alpha,
        form] if x is not None]))

    save_path = os.path.join(vis_dir, img_name + '.pdf')

    fig.savefig(save_path, dpi=600, transparent=True, bbox_inches=None, format='pdf')
    # plt.show()
    # print('(Axis 0) Elevation angle:', axs[0].elev, ' Azimuth angle:', axs[0].azim)
    # print('(Axis 1) Elevation angle:', axs[1].elev, ' Azimuth angle:', axs[1].azim)
    plt.close(fig)


def drop_and_truncate(
        arr,              # your decoding array, shape (..., n_trials, T_max)
        trial_lengths,    # 1-D array, len = n_trials
        drop_shortest_pct # e.g. 10  →  drop the shortest 10 %
):
    """
    1.  Drop trials whose length is below the requested percentile.
    2.  Optionally (see `crop`) crop the time axis so every kept trial
        is fully defined (no all-NaN tail).
    Returns
    -------
    arr_kept    : padded array with shorter trial axis, shape (..., n_keep, T_crop)
    lengths_new : updated 1-D array of trial lengths (n_keep,)
    """
    # ---------------------------------------------------------------
    # 1) decide which trials to keep
    thresh     = np.percentile(trial_lengths, drop_shortest_pct)
    keep_mask  = trial_lengths >= thresh                 # boolean, shape (n_trials,)
    lengths_new = trial_lengths[keep_mask]

    if keep_mask.sum() == 0:
        raise ValueError("Nothing left after dropping – lower your drop_shortest_pct")

    # ---------------------------------------------------------------
    # 2) keep only the selected trials
    arr_kept = arr[..., keep_mask, :]                    # (..., n_keep, T_max)

    # ---------------------------------------------------------------
    # 3) truncate to the minimum length of the kept trials
    T_crop   = lengths_new.min()
    arr_kept = arr_kept[..., :T_crop]                    # (..., n_keep, T_crop)

    return arr_kept, T_crop


def plot_state_timecourses(
        decoding,            # shape (n_states, n_trials, T)
                            #  → mean over axis=1 before plotting
        time_step   = 1.0,   #   seconds between samples; change if your bin width differs
        sem_band    = True,  #   draw ±SEM shading?
        title       = "",    #   panel title
        ylabel      = "decoding error",
        figsize     = (6, 4)):
    """
    Draw one line per continuous state, averaged over trials.

    - `decoding`   : ndarray, (n_states, n_trials, T)
    - `time_step`  : abscissa spacing (e.g. 0.02 → 50 Hz bins)
    """
    n_states, n_trials, T = decoding.shape
    t_axis = np.arange(T) * time_step

    means = np.nanmean(decoding, axis=1)               # (n_states, T)
    sems  = np.nanstd(decoding, axis=1, ddof=1) / np.sqrt(n_trials)  # (n_states, T)

    fig, ax = plt.subplots(figsize=figsize)

    for k in range(n_states):
        ax.plot(t_axis, means[k], label=f"state {k+1}")
        if sem_band:
            ax.fill_between(t_axis,
                            means[k] - sems[k],
                            means[k] + sems[k],
                            alpha=0.3)                 # default colour’s alpha only

    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_per_time_decoding_results_avg_session(
    unit_filter,
    input_unit_filter,
    data_format,
    label_format,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    truncate_percentile=10,
    visual_delay_time=0):

    print('Plotting decoding results...')

    rSLDS_same_speed_decoding_name = '_'.join(map(str, [x for x in [
        'decoding_totf',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        label_format,
        trial_filters,
        'same_speed',
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))
    
    rSLDS_cross_speed_decoding_name = '_'.join(map(str, [x for x in [
        'decoding_totf',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        label_format,
        trial_filters,
        'joint',
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))

    rSLDS_same_speed_decoding_slow_all = []
    rSLDS_same_speed_decoding_fast_all = []
    rSLDS_cross_speed_decoding_slow_all = []
    rSLDS_cross_speed_decoding_fast_all = []
    trial_length_slow_min_all = []
    trial_length_fast_min_all = []

    for session_data_name in session_data_names:

        session_results_dir = os.path.join(results_dir, session_data_name)

        rSLDS_same_speed_decoding = np.load(os.path.join(session_results_dir, rSLDS_same_speed_decoding_name + '.npz'))
        rSLDS_same_speed_decoding_slow = rSLDS_same_speed_decoding['decoding_errors_slow_test_per_time']
        rSLDS_same_speed_decoding_fast = rSLDS_same_speed_decoding['decoding_errors_fast_test_per_time']
        trial_lengths_slow_same_speed  = rSLDS_same_speed_decoding['trial_lengths_slow']
        trial_lengths_fast_same_speed  = rSLDS_same_speed_decoding['trial_lengths_fast']

        rSLDS_cross_speed_decoding = np.load(os.path.join(session_results_dir, rSLDS_cross_speed_decoding_name + '.npz'))
        rSLDS_cross_speed_decoding_slow = rSLDS_cross_speed_decoding['decoding_errors_slow_test_per_time']
        rSLDS_cross_speed_decoding_fast = rSLDS_cross_speed_decoding['decoding_errors_fast_test_per_time']
        trial_lengths_slow_cross_speed  = rSLDS_cross_speed_decoding['trial_lengths_slow']
        trial_lengths_fast_cross_speed  = rSLDS_cross_speed_decoding['trial_lengths_fast']

        assert np.array_equal(trial_lengths_slow_same_speed, trial_lengths_slow_cross_speed)
        assert np.array_equal(trial_lengths_fast_same_speed, trial_lengths_fast_cross_speed)
        trial_lengths_slow = trial_lengths_slow_same_speed
        trial_lengths_fast = trial_lengths_fast_same_speed

        # Take the first discrete state (LDS) and last n_iters
        rSLDS_same_speed_decoding_slow = rSLDS_same_speed_decoding_slow[:, :, 0, -1]
        rSLDS_same_speed_decoding_fast = rSLDS_same_speed_decoding_fast[:, :, 0, -1]
        rSLDS_cross_speed_decoding_slow = rSLDS_cross_speed_decoding_slow[:, :, 0, -1]
        rSLDS_cross_speed_decoding_fast = rSLDS_cross_speed_decoding_fast[:, :, 0, -1]

        # Truncate the decoding results with specified percentile
        rSLDS_same_speed_decoding_slow, trial_length_slow_min = drop_and_truncate(rSLDS_same_speed_decoding_slow, trial_lengths_slow, truncate_percentile)
        rSLDS_same_speed_decoding_fast, trial_length_fast_min = drop_and_truncate(rSLDS_same_speed_decoding_fast, trial_lengths_fast, truncate_percentile)
        rSLDS_cross_speed_decoding_slow, _ = drop_and_truncate(rSLDS_cross_speed_decoding_slow, trial_lengths_slow, truncate_percentile)
        rSLDS_cross_speed_decoding_fast, _ = drop_and_truncate(rSLDS_cross_speed_decoding_fast, trial_lengths_fast, truncate_percentile)

        trial_length_slow_min_all.append(trial_length_slow_min)
        trial_length_fast_min_all.append(trial_length_fast_min)

        # Average over random states
        rSLDS_same_speed_decoding_slow = np.mean(rSLDS_same_speed_decoding_slow, axis=0)
        rSLDS_same_speed_decoding_fast = np.mean(rSLDS_same_speed_decoding_fast, axis=0)
        rSLDS_cross_speed_decoding_slow = np.mean(rSLDS_cross_speed_decoding_slow, axis=0)
        rSLDS_cross_speed_decoding_fast = np.mean(rSLDS_cross_speed_decoding_fast, axis=0)

        # Append to all sessions
        rSLDS_same_speed_decoding_slow_all.append(rSLDS_same_speed_decoding_slow)
        rSLDS_same_speed_decoding_fast_all.append(rSLDS_same_speed_decoding_fast)
        rSLDS_cross_speed_decoding_slow_all.append(rSLDS_cross_speed_decoding_slow)
        rSLDS_cross_speed_decoding_fast_all.append(rSLDS_cross_speed_decoding_fast)

    ## Combine all sessions

    # Get the minimum trial length across all sessions
    trial_length_slow_min = np.min(trial_length_slow_min_all)
    trial_length_fast_min = np.min(trial_length_fast_min_all)

    # Get the number of trials across all sessions
    n_trials_slow_all = np.sum([x.shape[1] for x in rSLDS_same_speed_decoding_slow_all])
    n_trials_fast_all = np.sum([x.shape[1] for x in rSLDS_same_speed_decoding_fast_all])

    n_ns_states = len(ns_states)
    rSLDS_same_speed_decoding_slow_all_ = np.zeros((n_ns_states, n_trials_slow_all, trial_length_slow_min))
    rSLDS_same_speed_decoding_fast_all_ = np.zeros((n_ns_states, n_trials_fast_all, trial_length_fast_min))
    rSLDS_cross_speed_decoding_slow_all_ = np.zeros((n_ns_states, n_trials_slow_all, trial_length_slow_min))
    rSLDS_cross_speed_decoding_fast_all_ = np.zeros((n_ns_states, n_trials_fast_all, trial_length_fast_min))

    n_trials_slow_cumulative = 0
    n_trials_fast_cumulative = 0
    for i in range(len(session_data_names)):
        n_trials_slow = rSLDS_same_speed_decoding_slow_all[i].shape[1]
        n_trials_fast = rSLDS_same_speed_decoding_fast_all[i].shape[1]

        # Truncate to the minimum trial length
        rSLDS_same_speed_decoding_slow_all[i] = rSLDS_same_speed_decoding_slow_all[i][..., :trial_length_slow_min]
        rSLDS_same_speed_decoding_fast_all[i] = rSLDS_same_speed_decoding_fast_all[i][..., :trial_length_fast_min]
        rSLDS_cross_speed_decoding_slow_all[i] = rSLDS_cross_speed_decoding_slow_all[i][..., :trial_length_slow_min]
        rSLDS_cross_speed_decoding_fast_all[i] = rSLDS_cross_speed_decoding_fast_all[i][..., :trial_length_fast_min]

        # Concatenate all sessions
        rSLDS_same_speed_decoding_slow_all_[:, n_trials_slow_cumulative:n_trials_slow_cumulative + n_trials_slow, :] = rSLDS_same_speed_decoding_slow_all[i]
        rSLDS_same_speed_decoding_fast_all_[:, n_trials_fast_cumulative:n_trials_fast_cumulative + n_trials_fast, :] = rSLDS_same_speed_decoding_fast_all[i]
        rSLDS_cross_speed_decoding_slow_all_[:, n_trials_slow_cumulative:n_trials_slow_cumulative + n_trials_slow, :] = rSLDS_cross_speed_decoding_slow_all[i]
        rSLDS_cross_speed_decoding_fast_all_[:, n_trials_fast_cumulative:n_trials_fast_cumulative + n_trials_fast, :] = rSLDS_cross_speed_decoding_fast_all[i]

        n_trials_slow_cumulative += n_trials_slow
        n_trials_fast_cumulative += n_trials_fast

    decoding_results = {
        'fast':
        # 'near':
            {
                'LDS_mean_same_speed': rSLDS_same_speed_decoding_fast_all_,
                'LDS_mean_cross_speed': rSLDS_cross_speed_decoding_fast_all_,
                # 'LDS_mean_same_speed': rSLDS_same_speed_decoding_fast_all[0],
                # 'LDS_mean_cross_speed': rSLDS_cross_speed_decoding_fast_all[0],
            },
        'slow':
        # 'far':
            {
                'LDS_mean_same_speed': rSLDS_same_speed_decoding_slow_all_,
                'LDS_mean_cross_speed': rSLDS_cross_speed_decoding_slow_all_,
                # 'LDS_mean_same_speed': rSLDS_same_speed_decoding_slow_all[0],
                # 'LDS_mean_cross_speed': rSLDS_cross_speed_decoding_slow_all[0],
            }
    }

    # ipdb.set_trace()
    
    vmin = 0
    vmax = 90
    
    ## Plot results

    fig1 = plt.figure(figsize=(24, 12))
    ax0 = fig1.add_subplot(1, 2, 1, projection='3d')
    ax1 = fig1.add_subplot(1, 2, 2, projection='3d')
    axs = [ax0, ax1]

    for i_tf, trial_filter in enumerate(trial_filters):

        # Plot lines and fill under them
        for i_cs, n_states in enumerate(ns_states):

            color_list = ['blue', 'red']

            zs_list = [
                np.mean(decoding_results[trial_filter]['LDS_mean_same_speed'], axis=1)[i_cs],
                np.mean(decoding_results[trial_filter]['LDS_mean_cross_speed'], axis=1)[i_cs]
            ]

            for color, zs in zip(color_list, zs_list):

                trial_length = zs.shape[-1]

                xs = np.arange(trial_length)
                ys = np.full(trial_length, n_states)

                # Plot the line
                axs[i_tf].plot(
                    xs, 
                    ys, 
                    zs, 
                    color=color, 
                    lw=size_line,
                    alpha=alpha_line)

                # Create the polygon for filling
                verts = utils_vis.polygon_under_curve(xs, ys, zs, z_min=vmin)

                # Create a Poly3DCollection object
                poly = Poly3DCollection(
                    verts, 
                    facecolor=color, 
                    alpha=alpha_line_thin)

                # Add the polygon to the axes
                axs[i_tf].add_collection3d(poly)

        if trial_filter in ['slow', 'far']:
            xtick_step = 5
        else:
            xtick_step = 1

        axs[i_tf].set_xticks(xs[::xtick_step])
        axs[i_tf].set_xticklabels(np.round(xs[::xtick_step] * config.time_step + visual_delay_time, 2))
        axs[i_tf].set_yticks(ns_states)


        # Set labels and title
        axs[i_tf].set_xlabel('Time (s)', fontsize=label_fontsize)
        axs[i_tf].set_ylabel('# of Latent States', fontsize=label_fontsize)
        axs[i_tf].set_zlabel('Absolute Angle Error (degrees)', fontsize=label_fontsize)

        # axs[i_tf].set_xlim(ns_states[0], ns_states[-1])
        # axs[i_tf].set_ylim(ns_discrete_states[0], ns_discrete_states[-1])
        axs[i_tf].set_zlim(vmin, vmax)

        # axs[i_tf].set_yticklabels(['PCA'] + ns_discrete_states)

        # Adjust viewing angle
        # axs[i_tf].view_init(elev=10, azim=-75)
        axs[i_tf].view_init(elev=20, azim=-40)

        # axs[i_tf].tick_params(axis='both', which='major', labelsize=label_fontsize)

        ## Remove background panes
        axs[i_tf].xaxis.pane.fill = False
        axs[i_tf].yaxis.pane.fill = False
        axs[i_tf].zaxis.pane.fill = False

        ## Set grid colors
        # axs[i_tf].xaxis._axinfo['grid'].update(color='black')
        # axs[i_tf].yaxis._axinfo['grid'].update(color='black')
        # axs[i_tf].zaxis._axinfo['grid'].update(color='black')    

        # ## Create tick labels: label only at even indices, else empty string
        # tick_labels = [str(x) if i % 2 == 0 else '' for i, x in enumerate(ns_states)]

        # ## Set tick labels
        # axs[i_tf].set_xticklabels(tick_labels)


    ## Add custom legend
    # fig.subplots_adjust(bottom=0.2)  ## Increase the bottom margin

    # custom_lines = [
    #     Line2D([0], [0], color='blue',                   lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[0], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[1], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[2], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[3], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[4], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[5], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[6], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[7], lw=2, linestyle='-')]

    # legend_labels = ['PCA', '1', '2', '3', '4', '6', '8', '12', '16']

    # ## Create legend with custom elements
    # plt.legend(
    #     custom_lines, 
    #     legend_labels, 
    #     title='# of Discrete States',
    #     loc='upper center', 
    #     bbox_to_anchor=(-0.2, -0.15), 
    #     fancybox=False, 
    #     shadow=False, 
    #     ncol=9)

    # plt.show()

    ## Write image
    if len(session_data_names) > 1:
        session_data_names_str = str(len(session_data_names)) + '_sessions'
    else:
        session_data_names_str = str(session_data_names)

    img_name = '_'.join(map(str, [x for x in [
        session_data_names_str,
        'per_time_decoding',
        truncate_percentile,
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        label_format,
        trial_filters,
        dynamics_class,
        emission_class,
        init_type,
        alpha] if x is not None]))

    save_path = os.path.join(vis_dir, img_name + '.pdf')

    fig1.savefig(save_path, dpi=600, transparent=True)
    # plt.show()
    # print('(Axis 0) Elevation angle:', axs[0].elev, ' Azimuth angle:', axs[0].azim)
    # print('(Axis 1) Elevation angle:', axs[1].elev, ' Azimuth angle:', axs[1].azim)
    plt.close(fig1)


def plot_inference_results_avg_session(
    unit_filter,
    input_unit_filter,
    data_format,
    train_test_option,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    train_or_test='test',
    inference_type='forecast',
    form='heatmap'):

    print('Plotting inference results...')

    task_name = session_data_names[0].split('_')[-1]

    xticks = np.arange(len(ns_states))
    xticklabels = ns_states

    rSLDS_inference_name = '_'.join(map(str, [x for x in [
        'inference',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        train_test_option,
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        alpha] if x is not None]))
    
    inference_baseline_name = '_'.join(map(str, [x for x in [
        'inference_baseline',
        unit_filter,
        data_format] if x is not None]))

    rSLDS_r2_forecast_slow_all  = []
    rSLDS_r2_forecast_fast_all  = []
    # r2_forecast_baseline_slow_all = []
    # r2_forecast_baseline_fast_all = []

    inference_name = 'r2_' + inference_type

    for session_data_name in session_data_names:

        session_results_dir = os.path.join(results_dir, session_data_name)

        rSLDS_inference = np.load(os.path.join(session_results_dir, rSLDS_inference_name + '.npz'))
        rSLDS_r2_forecast_slow = rSLDS_inference[inference_name + '_slow']
        rSLDS_r2_forecast_fast = rSLDS_inference[inference_name + '_fast']
        # inference_baseline = np.load(os.path.join(session_results_dir, inference_baseline_name + '.npz'))
        # r2_forecast_baseline_slow = inference_baseline['r2_forecast_baseline_slow']
        # r2_forecast_baseline_fast = inference_baseline['r2_forecast_baseline_fast']
        ipdb.set_trace()

        ## Take the last n_iters, then add to all sessions
        rSLDS_r2_forecast_slow = rSLDS_r2_forecast_slow[..., -1]
        rSLDS_r2_forecast_fast = rSLDS_r2_forecast_fast[..., -1]
        # r2_forecast_baseline_slow = r2_forecast_baseline_slow
        # r2_forecast_baseline_fast = r2_forecast_baseline_fast

        rSLDS_r2_forecast_slow_all.append(rSLDS_r2_forecast_slow)
        rSLDS_r2_forecast_fast_all.append(rSLDS_r2_forecast_fast)
        # r2_forecast_baseline_slow_all.append(r2_forecast_baseline_slow)
        # r2_forecast_baseline_fast_all.append(r2_forecast_baseline_fast)

    ## Stack all sessions
    rSLDS_r2_forecast_slow_all  = np.stack(rSLDS_r2_forecast_slow_all,  axis=0)
    rSLDS_r2_forecast_fast_all  = np.stack(rSLDS_r2_forecast_fast_all,  axis=0)
    # r2_forecast_baseline_slow_all = np.stack(r2_forecast_baseline_slow_all, axis=0)
    # r2_forecast_baseline_fast_all = np.stack(r2_forecast_baseline_fast_all, axis=0)

    ## Compute mean and standard error over sessions, random states, and folds
    rSLDS_r2_forecast_slow_all_mean = np.mean(rSLDS_r2_forecast_slow_all, axis=(0, 2, 3))
    rSLDS_r2_forecast_fast_all_mean = np.mean(rSLDS_r2_forecast_fast_all, axis=(0, 2, 3))
    # r2_forecast_baseline_slow_all_mean = np.mean(r2_forecast_baseline_slow_all, axis=(0, 2))
    # r2_forecast_baseline_fast_all_mean = np.mean(r2_forecast_baseline_fast_all, axis=(0, 2))

    correction_factor = np.sqrt(len(session_data_names) * len(random_states) * n_folds)
    rSLDS_r2_forecast_slow_all_se = np.std(rSLDS_r2_forecast_slow_all, axis=(0, 2, 3)) / correction_factor
    rSLDS_r2_forecast_fast_all_se = np.std(rSLDS_r2_forecast_fast_all, axis=(0, 2, 3)) / correction_factor
    # r2_forecast_baseline_slow_all_se = np.std(r2_forecast_baseline_slow_all, axis=(0, 2)) / correction_factor
    # r2_forecast_baseline_fast_all_se = np.std(r2_forecast_baseline_fast_all, axis=(0, 2)) / correction_factor


    inference_results = {
        'slow':
        # 'far':
            {
                'mean': rSLDS_r2_forecast_slow_all_mean,
                'se': rSLDS_r2_forecast_slow_all_se,
            },
        'fast':
        # 'near':
            {
                'mean': rSLDS_r2_forecast_fast_all_mean,
                'se': rSLDS_r2_forecast_fast_all_se,
            }
    }

    vmin = 0.5
    vmax = 1

    if train_or_test == 'train':
        train_or_test_id = 0
    elif train_or_test == 'test':
        train_or_test_id = 1

    ## Plot results
    if form == 'heatmap':

        fig, axs = plt.subplots(1, 2, figsize=(45*mm, 22.5*mm), sharey=True)

        for i_tf, trial_filter in enumerate(trial_filters):

            sns.heatmap(
                inference_results[trial_filter]['mean'][train_or_test_id, ...].T,
                ax=axs[i_tf],
                vmin=vmin,
                vmax=vmax,
                cmap='rocket',
                cbar=False,          # ← suppress the color-bar
                # square=True,         # ← make the axes a true square
                xticklabels=False,   # ← drop x-axis tick labels
                yticklabels=False,   # ← drop y-axis tick labels
                linewidths=0,
                linecolor=None,
                rasterized=True,
            )
        
            # If you also want to hide the tick marks themselves:
            axs[i_tf].tick_params(left=False, bottom=False)
            axs[i_tf].set_box_aspect(1)

            # axs[i_tf].set_title('Variance explained (R²)', fontsize=5)
            # axs[i_tf].set_xticklabels(['LDS'] + ns_discrete_states)
            # axs[i_tf].set_xticklabels(ns_discrete_states)
            # axs[i_tf].set_yticklabels(ns_states)
            # axs[i_tf].set_xlabel('Number of discrete states', fontsize=5)
            # axs[i_tf].set_ylabel('Number of continuous states', fontsize=5)
            axs[i_tf].invert_yaxis()

        fig.tight_layout()

    elif form == 'waterfall':

        fig = plt.figure(figsize=(50*mm, 25*mm))
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        axs = [ax0, ax1]

        for i_tf, trial_filter in enumerate(trial_filters):

            # Plot lines and fill under them
            for i_cs in range(len(ns_states)):

                xs = np.arange(len(ns_discrete_states))
                ys = np.full_like(ns_discrete_states, i_cs)
                zs = inference_results[trial_filter]['mean'][train_or_test_id, i_cs]

                # Plot the line
                axs[i_tf].plot(
                    xs, 
                    ys, 
                    zs, 
                    color=continuous_state_colors[i_cs], 
                    lw=0.25,
                    alpha=alpha_line)

                # Create the polygon for filling
                verts = utils_vis.polygon_under_curve(xs, ys, zs, z_min=vmin)

                # Create a Poly3DCollection object
                poly = Poly3DCollection(
                    verts, 
                    facecolor=continuous_state_colors[i_cs], 
                    alpha=0.1)

                # Add the polygon to the axes
                axs[i_tf].add_collection3d(poly)

            # Set labels and title
            # axs[i_tf].set_ylabel('Number of continuous states', fontsize=5, labelpad=-8)
            # axs[i_tf].set_xlabel('Number of discrete states', fontsize=5, labelpad=-8)
            # axs[i_tf].set_zlabel('Variance explained (R²)', fontsize=5, labelpad=-8)

            # axs[i_tf].set_xlim(ns_states[0], ns_states[-1])
            # axs[i_tf].set_ylim(ns_discrete_states[0], ns_discrete_states[-1])
            axs[i_tf].set_zlim(vmin, vmax)

            axs[i_tf].set_yticks(np.arange(len(ns_states)))
            # axs[i_tf].set_yticklabels(ns_states)
            axs[i_tf].set_yticklabels([])
            axs[i_tf].set_xticks(np.arange(len(ns_discrete_states)))
            # axs[i_tf].set_xticklabels(ns_discrete_states)
            axs[i_tf].set_xticklabels([])

            axs[i_tf].set_zticklabels([])

            # Adjust viewing angle
            axs[i_tf].view_init(elev=20, azim=-40)

            # axs[i_tf].tick_params(axis='both', which='major', labelsize=label_fontsize)

            ## Remove background panes
            axs[i_tf].xaxis.pane.fill = False
            axs[i_tf].yaxis.pane.fill = False
            axs[i_tf].zaxis.pane.fill = False
            axs[i_tf].grid(False) 

            ## Set grid colors
            # axs[i_tf].xaxis._axinfo['grid'].update(color='black')
            # axs[i_tf].yaxis._axinfo['grid'].update(color='black')
            # axs[i_tf].zaxis._axinfo['grid'].update(color='black')    

            # ## Create tick labels: label only at even indices, else empty string
            # tick_labels = [str(x) if i % 2 == 0 else '' for i, x in enumerate(ns_states)]

            # # ## Set tick labels
            # axs[i_tf].set_yticklabels(tick_labels)




    ## Write image
    session_data_names_str = str(len(session_data_names)) + '_sessions'

    img_name = '_'.join(map(str, [x for x in [
        task_name,
        session_data_names_str,
        'inference',
        train_or_test,
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        train_test_option,
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha,
        inference_type,
        form] if x is not None]))

    save_path = os.path.join(vis_dir, img_name + '.pdf')

    fig.savefig(save_path, dpi=600, transparent=True, bbox_inches=None, format='pdf')
    plt.close(fig)


def plot_per_time_inference_results_avg_session(
    unit_filter,
    input_unit_filter,
    data_format,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    inference_type='forecast',
    truncate_percentile=10,
    visual_delay_time=0):

    print('Plotting inference results...')

    rSLDS_same_speed_inference_name = '_'.join(map(str, [x for x in [
        'inference',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        'same_speed',
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))
    
    rSLDS_cross_speed_inference_name = '_'.join(map(str, [x for x in [
        'inference',
        unit_filter,
        # input_unit_filter,
        None,
        window_config,
        # time_offset,
        None,
        data_format,
        trial_filters,
        # 'cross_speed',
        'same_speed',
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))

    rSLDS_same_speed_inference_slow_all = []
    rSLDS_same_speed_inference_fast_all = []
    rSLDS_cross_speed_inference_slow_all = []
    rSLDS_cross_speed_inference_fast_all = []
    trial_length_slow_min_all = []
    trial_length_fast_min_all = []

    inference_name = 'r2_' + inference_type

    for session_data_name in session_data_names:

        session_results_dir = os.path.join(results_dir, session_data_name)

        rSLDS_same_speed_inference = np.load(os.path.join(session_results_dir, rSLDS_same_speed_inference_name + '.npz'))
        rSLDS_same_speed_inference_slow = rSLDS_same_speed_inference[inference_name + '_slow_test_per_time']
        rSLDS_same_speed_inference_fast = rSLDS_same_speed_inference[inference_name + '_fast_test_per_time']
        trial_lengths_slow_same_speed = rSLDS_same_speed_inference['trial_lengths_slow'] - 1
        trial_lengths_fast_same_speed = rSLDS_same_speed_inference['trial_lengths_fast'] - 1

        rSLDS_cross_speed_inference = np.load(os.path.join(session_results_dir, rSLDS_cross_speed_inference_name + '.npz'))
        rSLDS_cross_speed_inference_slow = rSLDS_cross_speed_inference[inference_name + '_slow_test_per_time']
        rSLDS_cross_speed_inference_fast = rSLDS_cross_speed_inference[inference_name + '_fast_test_per_time']
        trial_lengths_slow_cross_speed = rSLDS_cross_speed_inference['trial_lengths_slow'] - 1
        trial_lengths_fast_cross_speed = rSLDS_cross_speed_inference['trial_lengths_fast'] - 1

        assert np.array_equal(trial_lengths_slow_same_speed, trial_lengths_slow_cross_speed)
        assert np.array_equal(trial_lengths_fast_same_speed, trial_lengths_fast_cross_speed)
        trial_lengths_slow = trial_lengths_slow_same_speed
        trial_lengths_fast = trial_lengths_fast_same_speed

        # Take the first discrete state (LDS) and last n_iters
        rSLDS_same_speed_inference_slow = rSLDS_same_speed_inference_slow[:, :, 0, -1]
        rSLDS_same_speed_inference_fast = rSLDS_same_speed_inference_fast[:, :, 0, -1]
        rSLDS_cross_speed_inference_slow = rSLDS_cross_speed_inference_slow[:, :, 0, -1]
        rSLDS_cross_speed_inference_fast = rSLDS_cross_speed_inference_fast[:, :, 0, -1]

        # Truncate the inference results with specified percentile
        rSLDS_same_speed_inference_slow, trial_length_slow_min = drop_and_truncate(rSLDS_same_speed_inference_slow, trial_lengths_slow, truncate_percentile)
        rSLDS_same_speed_inference_fast, trial_length_fast_min = drop_and_truncate(rSLDS_same_speed_inference_fast, trial_lengths_fast, truncate_percentile)
        rSLDS_cross_speed_inference_slow, _ = drop_and_truncate(rSLDS_cross_speed_inference_slow, trial_lengths_slow, truncate_percentile)
        rSLDS_cross_speed_inference_fast, _ = drop_and_truncate(rSLDS_cross_speed_inference_fast, trial_lengths_fast, truncate_percentile)

        trial_length_slow_min_all.append(trial_length_slow_min)
        trial_length_fast_min_all.append(trial_length_fast_min)

        # Average over random states
        rSLDS_same_speed_inference_slow = np.mean(rSLDS_same_speed_inference_slow, axis=0)
        rSLDS_same_speed_inference_fast = np.mean(rSLDS_same_speed_inference_fast, axis=0)
        rSLDS_cross_speed_inference_slow = np.mean(rSLDS_cross_speed_inference_slow, axis=0)
        rSLDS_cross_speed_inference_fast = np.mean(rSLDS_cross_speed_inference_fast, axis=0)

        # Append to all sessions
        rSLDS_same_speed_inference_slow_all.append(rSLDS_same_speed_inference_slow)
        rSLDS_same_speed_inference_fast_all.append(rSLDS_same_speed_inference_fast)
        rSLDS_cross_speed_inference_slow_all.append(rSLDS_cross_speed_inference_slow)
        rSLDS_cross_speed_inference_fast_all.append(rSLDS_cross_speed_inference_fast)

    ## Combine all sessions

    # Get the minimum trial length across all sessions
    trial_length_slow_min = np.min(trial_length_slow_min_all)
    trial_length_fast_min = np.min(trial_length_fast_min_all)

    # Get the number of trials across all sessions
    n_trials_slow_all = np.sum([x.shape[1] for x in rSLDS_same_speed_inference_slow_all])
    n_trials_fast_all = np.sum([x.shape[1] for x in rSLDS_same_speed_inference_fast_all])

    n_ns_states = len(ns_states)
    rSLDS_same_speed_inference_slow_all_ = np.zeros((n_ns_states, n_trials_slow_all, trial_length_slow_min))
    rSLDS_same_speed_inference_fast_all_ = np.zeros((n_ns_states, n_trials_fast_all, trial_length_fast_min))
    rSLDS_cross_speed_inference_slow_all_ = np.zeros((n_ns_states, n_trials_slow_all, trial_length_slow_min))
    rSLDS_cross_speed_inference_fast_all_ = np.zeros((n_ns_states, n_trials_fast_all, trial_length_fast_min))

    n_trials_slow_cumulative = 0
    n_trials_fast_cumulative = 0
    for i in range(len(session_data_names)):
        n_trials_slow = rSLDS_same_speed_inference_slow_all[i].shape[1]
        n_trials_fast = rSLDS_same_speed_inference_fast_all[i].shape[1]

        # Truncate to the minimum trial length
        rSLDS_same_speed_inference_slow_all[i] = rSLDS_same_speed_inference_slow_all[i][..., :trial_length_slow_min]
        rSLDS_same_speed_inference_fast_all[i] = rSLDS_same_speed_inference_fast_all[i][..., :trial_length_fast_min]
        rSLDS_cross_speed_inference_slow_all[i] = rSLDS_cross_speed_inference_slow_all[i][..., :trial_length_slow_min]
        rSLDS_cross_speed_inference_fast_all[i] = rSLDS_cross_speed_inference_fast_all[i][..., :trial_length_fast_min]

        # Concatenate all sessions
        rSLDS_same_speed_inference_slow_all_[:, n_trials_slow_cumulative:n_trials_slow_cumulative + n_trials_slow, :] = rSLDS_same_speed_inference_slow_all[i]
        rSLDS_same_speed_inference_fast_all_[:, n_trials_fast_cumulative:n_trials_fast_cumulative + n_trials_fast, :] = rSLDS_same_speed_inference_fast_all[i]
        rSLDS_cross_speed_inference_slow_all_[:, n_trials_slow_cumulative:n_trials_slow_cumulative + n_trials_slow, :] = rSLDS_cross_speed_inference_slow_all[i]
        rSLDS_cross_speed_inference_fast_all_[:, n_trials_fast_cumulative:n_trials_fast_cumulative + n_trials_fast, :] = rSLDS_cross_speed_inference_fast_all[i]

        n_trials_slow_cumulative += n_trials_slow
        n_trials_fast_cumulative += n_trials_fast

    inference_results = {
        'fast':
        # 'near':
            {
                'LDS_mean_same_speed': rSLDS_same_speed_inference_fast_all_,
                'LDS_mean_cross_speed': rSLDS_cross_speed_inference_fast_all_,
                # 'LDS_mean_same_speed': rSLDS_same_speed_inference_fast_all[0],
                # 'LDS_mean_cross_speed': rSLDS_cross_speed_inference_fast_all[0],
            },
        'slow':
        # 'far':
            {
                'LDS_mean_same_speed': rSLDS_same_speed_inference_slow_all_,
                'LDS_mean_cross_speed': rSLDS_cross_speed_inference_slow_all_,
                # 'LDS_mean_same_speed': rSLDS_same_speed_inference_slow_all[0],
                # 'LDS_mean_cross_speed': rSLDS_cross_speed_inference_slow_all[0],
            }
    }
    
    vmin = 0
    vmax = 1
    
    ## Plot results

    fig1 = plt.figure(figsize=(90*mm, 45*mm))
    ax0 = fig1.add_subplot(1, 2, 1, projection='3d')
    ax1 = fig1.add_subplot(1, 2, 2, projection='3d')
    axs1 = [ax0, ax1]

    fig2, axs2 = plt.subplots(1, 2, figsize=(90*mm, 40*mm), sharey=True)


    for i_tf, trial_filter in enumerate(trial_filters):

        # Plot lines and fill under them
        for i_cs, n_states in enumerate(ns_states):

            # Plot fig1
            # Only plot even numbered states
            if i_cs % 2 == 1:
                continue
            linestyles = ['-', '--']

            zs_list = [
                np.mean(inference_results[trial_filter]['LDS_mean_same_speed'],  axis=1)[i_cs],
                np.mean(inference_results[trial_filter]['LDS_mean_cross_speed'], axis=1)[i_cs]
            ]

            trial_length = zs_list[0].shape[-1]

            xs = np.arange(trial_length)
            ys = np.full(trial_length, n_states)

            for linestyle, zs in zip(linestyles, zs_list):

                # Plot the line
                axs1[i_tf].plot(
                    xs, 
                    ys, 
                    zs, 
                    color=continuous_state_colors[i_cs], 
                    lw=0.25,
                    alpha=alpha_line_thick,
                    linestyle=linestyle)

                # Create the polygon for filling
                verts = utils_vis.polygon_under_curve(xs, ys, zs, z_min=vmin)

                # Create a Poly3DCollection object
                poly = Poly3DCollection(
                    verts, 
                    facecolor=continuous_state_colors[i_cs], 
                    alpha=0.1)

                # Add the polygon to the axes
                axs1[i_tf].add_collection3d(poly)

            
            # Plot fig2
            diffs = inference_results[trial_filter]['LDS_mean_same_speed'][i_cs] - inference_results[trial_filter]['LDS_mean_cross_speed'][i_cs]
            diffs_mean = np.mean(diffs, axis=0)
            diffs_sem = np.std(diffs, axis=0, ddof=1) / np.sqrt(diffs.shape[0])

            # Plot the line
            # axs2[i_tf].plot(
            #     xs,
            #     diffs_mean,
            #     color=continuous_state_colors[i_cs],
            #     lw=size_line_thin,
            #     alpha=alpha_line)
    
            axs2[i_tf].fill_between(
                xs, 
                diffs_mean - diffs_sem, 
                diffs_mean + diffs_sem,
                color=continuous_state_colors[i_cs], 
                alpha=alpha_line, 
                linewidth=0)


        if trial_filter in ['slow', 'far']:
            xtick_step = 0.5
        else:
            # xtick_step = 0.1
            xtick_step = 0.2



        

        # # ────────────────────────────────────────────────────────────────
        # # --- SETUP — you already have these three numbers
        # # ────────────────────────────────────────────────────────────────
        # dt   = config.time_step        # seconds per sample            e.g. 0.01
        # t0   = visual_delay_time       # first sample’s absolute time  e.g. 0.139
        # T    = trial_length            # number of samples             e.g. 30

        # # ────────────────────────────────────────────────────────────────
        # # 1. choose a “nice” step in seconds and build the tick list
        # # ────────────────────────────────────────────────────────────────
        # t_last    = t0 + (T-1)*dt                            # last time-point
        # tick_times = np.arange(t0, t_last + 1e-9, xtick_step) # t0, t0+0.05, …

        # # ────────────────────────────────────────────────────────────────
        # # 2. convert those times back to *index* positions (data coords)
        # #    position_i = (time_i − t0) / dt
        # # ────────────────────────────────────────────────────────────────
        # tick_idx = (tick_times - t0) / dt

        # # ────────────────────────────────────────────────────────────────
        # # 3. set ticks + labels on each axis
        # # ────────────────────────────────────────────────────────────────
        # for ax in (axs1[i_tf], axs2[i_tf]):
        #     ax.set_xticks(tick_idx)
        #     ax.set_xticklabels([f'{t:.2f}' for t in tick_times])




        # ───────────── parameters you already know ─────────────
        dt   = config.time_step          # seconds per sample, e.g. 0.01
        t0   = visual_delay_time         # first sample’s absolute time, e.g. 0.139
        T    = trial_length              # number of samples

        # ───────────── 1. build the tick-time list ─────────────
        t_last      = t0 + (T - 1) * dt
        first_nice  = (np.ceil(t0 / xtick_step)) * xtick_step      # 0.150 in the example
        tick_times  = np.concatenate([[t0],
                                      np.arange(first_nice, t_last + 1e-9, xtick_step)])
        # Guard against a duplicate when t0 is itself “nice”
        tick_times  = np.unique(np.round(tick_times, 10))

        # ───────────── 2. convert times → sample indices ───────
        tick_idx = (tick_times - t0) / dt     # index 0 for t0, >0 for the rest

        # ───────────── 3. apply to every axis you care about ───
        for ax in (axs1[i_tf], axs2[i_tf]):
            ax.set_xticks(tick_idx)
            ax.set_xticklabels([f'{tick_times[0]:.2f}'] + [f'{t:.1f}' for t in tick_times[1:]])





        # Set labels and title
        axs1[i_tf].set_xlabel('Time (s)', fontsize=5, labelpad=-8)
        axs1[i_tf].set_ylabel('Number of latent states', fontsize=5, labelpad=-8)
        

        axs2[i_tf].set_xlabel('Time (s)', fontsize=5)

        # axs1[i_tf].set_xlim(ns_states[0], ns_states[-1])
        # axs1[i_tf].set_ylim(ns_discrete_states[0], ns_discrete_states[-1])
        axs1[i_tf].set_zlim(vmin, vmax)
        # axs1[i_tf].set_ylim(ns_states[0])
        axs1[i_tf].set_xlim(tick_idx[0])

        axs2[i_tf].set_ylim(-0.2, 0.2)
        axs2[i_tf].set_xlim(tick_idx[0])

        axs2[i_tf].axhline(0, color='black', lw=0.25, ls='--', alpha=alpha_line)
        
        axs1[i_tf].set_yticks(np.arange(ns_states[0], ns_states[-1] + 1, 4))
        

        # Adjust viewing angle
        # axs1[i_tf].view_init(elev=10, azim=-75)
        # axs1[i_tf].view_init(elev=20, azim=-40)
        axs1[i_tf].view_init(elev=25, azim=-130)

        # axs1[i_tf].tick_params(axis='both', which='major', labelsize=5)

        ## Remove background panes
        axs1[i_tf].xaxis.pane.fill = False
        axs1[i_tf].yaxis.pane.fill = False
        axs1[i_tf].zaxis.pane.fill = False
        axs1[i_tf].grid(False) 

        # ► put this immediately after you create each 3-D axis
        for axis in (axs1[i_tf].xaxis, axs1[i_tf].yaxis, axs1[i_tf].zaxis):
            # 1) bring tick LABELS closer to their axis lines
            axis.set_tick_params(pad=-4)      # default ~4–6 → try 0–2

            # # 2) shorten the little tick MARKS themselves (optional)
            # axis.set_tick_params(length=2)   # default 3.5

            # axis._axinfo['tick']['pad']   = 0   # label ↔ axis distance (pts)
            # axis._axinfo['tick']['outward_factor'] = 0.2  # tick length scaling


        axs2[i_tf].spines[['right', 'top']].set_visible(False)  # keep bottom spine
        

        ## Set grid colors
        # axs1[i_tf].xaxis._axinfo['grid'].update(color='black')
        # axs1[i_tf].yaxis._axinfo['grid'].update(color='black')
        # axs1[i_tf].zaxis._axinfo['grid'].update(color='black')    

        # ## Create tick labels: label only at even indices, else empty string
        # tick_labels = [str(x) if i % 2 == 0 else '' for i, x in enumerate(ns_states)]

        # ## Set tick labels
        # axs1[i_tf].set_xticklabels(tick_labels)


        ## Significance analysis
        same = inference_results[trial_filter]['LDS_mean_same_speed']
        cross = inference_results[trial_filter]['LDS_mean_cross_speed']
        diff = same - cross                       # positive ⇒ same-speed better

        n_states, _, T = diff.shape
        p_raw = np.ones((n_states, T))

        for k in range(n_states):
            for t in range(T):
                # Wilcoxon signed-rank on the trials of that state & time bin
                _, p = wilcoxon(diff[k, :, t], alternative='greater')  # “greater” = same>cross
                p_raw[k, t] = p

        # FDR-correct across the whole matrix
        rej, p_fdr = fdrcorrection(p_raw.ravel(), alpha=0.05)
        p_fdr = p_fdr.reshape(p_raw.shape)
        sig = p_fdr < 0.05

        def significant_segments(mask, xs):
            """Return list of (x_start, x_end) intervals where mask==True."""
            if mask.ndim != 1:
                raise ValueError("mask must be 1-D along time")
            # find rising & falling edges
            edges = np.diff(mask.astype(int), prepend=0, append=0)
            starts = np.where(edges ==  1)[0]
            ends   = np.where(edges == -1)[0]    # first False AFTER streak
            return [(xs[s], xs[e-1]) for s, e in zip(starts, ends)]
    
        # after axs2[i_tf] has been created
        divider = make_axes_locatable(axs2[i_tf])
        ax_sig  = divider.append_axes("top", size="15%", pad=0.05, sharex=axs2[i_tf])

        # no y-ticks / labels
        ax_sig.yaxis.set_visible(False)
        ax_sig.xaxis.set_visible(False)
        ax_sig.spines[['bottom', 'right', 'left', 'top']].set_visible(False)  # keep bottom spine
        # ax_sig.spines[['right', 'left', 'top']].set_visible(False)  # keep bottom spine

        # --------------------------------------------------------------
        # Config – tweak these two numbers until it feels airy enough
        # --------------------------------------------------------------
        row_height = 1.4      # 1.0 = bars touch; >1 = extra gap between bars
        edge_pad   = 0      # extra empty space above the top bar & below the bottom

        # --------------------------------------------------------------
        #  A. draw the stacked bars with more head-room
        # --------------------------------------------------------------
        for j, n_state in enumerate(ns_states):
            if n_state % 2 == 1:  # only plot even numbered states
                continue
            y_level = (j + 0.5) * row_height          # centre of that bar’s lane
            for x0, x1 in significant_segments(sig[j], xs):
                ax_sig.hlines(y_level, x0, x1,
                              color=continuous_state_colors[j], lw=1, alpha=alpha_line_thick)

        # --------------------------------------------------------------
        #  B.  set the y-axis limits so bars don’t hug the panel edges
        # --------------------------------------------------------------
        n_rows = len(ns_states)
        ax_sig.set_ylim(-edge_pad, n_rows*row_height + edge_pad)


    axs1[0].set_zlabel('Variance explained (R²)', fontsize=5, labelpad=-8)
    axs2[0].set_ylabel('Difference in variance explained (R²)', fontsize=5)


    ## Add custom legend
    # handles = [mpatches.Patch(color=continuous_state_colors[i], label=f'{s}')
    #        for i, s in enumerate(ns_states)]

    # n_cols = 20                           # wrap to keep it compact
    # axs2[0].legend(
    #     handles,
    #     [str(s) for s in ns_states],
    #     title="# latent states",
    #     loc="lower center",
    #     bbox_to_anchor=(1.1, -0.3),     # (x-centre, y) in figure coords
    #     ncols=n_cols,             # wrap to keep it compact
    #     # frameon=False,
    #     handlelength=0.9, handleheight=0.9,
    #     columnspacing=0.8, borderpad=0.4,
    #     fontsize=5, 
    #     title_fontsize=5,
    # )

    # (optional) tighten layout so the legend isn't cut off when you save
    # fig2.subplots_adjust(bottom=0.2)      # tweak as needed

    fig2.tight_layout()

    ## Write image
    if len(session_data_names) > 1:
        session_data_names_str = str(len(session_data_names)) + '_sessions'
    else:
        session_data_names_str = str(session_data_names)

    img_name = '_'.join(map(str, [x for x in [
        session_data_names_str,
        'per_time_inference',
        inference_type,
        truncate_percentile,
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        dynamics_class,
        emission_class,
        init_type,
        alpha] if x is not None]))

    save_path1 = os.path.join(vis_dir, img_name + '.pdf')
    save_path2 = os.path.join(vis_dir, img_name + '_diff.pdf')

    fig1.savefig(save_path1, dpi=600, transparent=True, bbox_inches=None, format='pdf')
    fig2.savefig(save_path2, dpi=600, transparent=True, bbox_inches=None, format='pdf')
    # plt.show()
    print('(Axis 0) Elevation angle:', axs1[0].elev, ' Azimuth angle:', axs1[0].azim)
    print('(Axis 1) Elevation angle:', axs1[1].elev, ' Azimuth angle:', axs1[1].azim)
    plt.close(fig1)
    plt.close(fig2)
    

    # Create a continuous colormap from seaborn's 'rocket' palette
    cmap = sns.color_palette("rocket", as_cmap=True)
    norm = Normalize(vmin=2, vmax=20)

    # Build the figure
    fig, ax = plt.subplots(figsize=(20*mm, 2*mm))
    # fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Add a horizontal colorbar without ticks or labels
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='horizontal'
    )
    cbar.set_ticks([])
    cbar.outline.set_visible(False)

    # Remove the axes frame completely
    ax.set_frame_on(False)
    fig.savefig(os.path.join(vis_dir, 'continuous_states_color_bar.pdf'), bbox_inches=None, transparent=True, dpi=600, format='pdf')


def plot_dsupr_results_avg_session(
    unit_filter,
    input_unit_filter,
    data_format,
    train_test_option,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    train_or_test='test',
    form='lines'):

    print('Plotting DSUP Ratio results...')

    task_name = session_data_names[0].split('_')[-1]

    xticks = np.arange(len(ns_states))
    xticklabels = ns_states

    rSLDS_dsupr_name = '_'.join(map(str, [x for x in [
        'dsupr',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        train_test_option,
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))

    rSLDS_dsupr_slow_all = []
    rSLDS_dsupr_fast_all = []

    for session_data_name in session_data_names:

        session_results_dir = os.path.join(results_dir, session_data_name)

        rSLDS_dsupr = np.load(os.path.join(session_results_dir, rSLDS_dsupr_name + '.npz'))
        rSLDS_dsupr_slow = rSLDS_dsupr['dsupr_slow']
        rSLDS_dsupr_fast = rSLDS_dsupr['dsupr_fast']

        ## Take the last n_iters, then add to all sessions
        rSLDS_dsupr_slow = rSLDS_dsupr_slow[..., -1]
        rSLDS_dsupr_fast = rSLDS_dsupr_fast[..., -1]

        rSLDS_dsupr_slow_all.append(rSLDS_dsupr_slow)
        rSLDS_dsupr_fast_all.append(rSLDS_dsupr_fast)

    ## Stack all sessions
    rSLDS_dsupr_slow_all = np.stack(rSLDS_dsupr_slow_all, axis=0)
    rSLDS_dsupr_fast_all = np.stack(rSLDS_dsupr_fast_all, axis=0)

    ## Compute mean and standard error over sessions, random states, and folds
    rSLDS_dsupr_slow_all_mean = np.mean(rSLDS_dsupr_slow_all, axis=(0, 2, 3))
    rSLDS_dsupr_fast_all_mean = np.mean(rSLDS_dsupr_fast_all, axis=(0, 2, 3))

    correction_factor = np.sqrt(len(session_data_names) * len(random_states) * n_folds)
    rSLDS_dsupr_slow_all_se = np.std(rSLDS_dsupr_slow_all, axis=(0, 2, 3)) / correction_factor
    rSLDS_dsupr_fast_all_se = np.std(rSLDS_dsupr_fast_all, axis=(0, 2, 3)) / correction_factor

    dsupr_results = {
        # 'slow':
        'far':
            {
                'mean': rSLDS_dsupr_slow_all_mean,
                'se': rSLDS_dsupr_slow_all_se,
            },
        # 'fast':
        'near':
            {
                'mean': rSLDS_dsupr_fast_all_mean,
                'se': rSLDS_dsupr_fast_all_se,
            }
    }

    vmin = 0
    vmax = 0.6

    if train_or_test == 'train':
        train_or_test_id = 0
    elif train_or_test == 'test':
        train_or_test_id = 1

    ## Plot results
    if form == 'heatmap':

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        for i_tf, trial_filter in enumerate(trial_filters):

            sns.heatmap(
                dsupr_results[trial_filter]['mean'][train_or_test_id, ...], 
                ax=axs[i_tf], 
                cbar=True,
                vmin=vmin,
                vmax=vmax,
                cmap='viridis')

            axs[i_tf].set_title('DSUP Ratio')
            # axs[i_tf].set_xticklabels(['LDS'] + ns_discrete_states)
            axs[i_tf].set_xticklabels(ns_discrete_states)
            axs[i_tf].set_yticklabels(ns_states)
            axs[i_tf].set_xlabel('# of Discrete States')
            axs[i_tf].set_ylabel('# of Continuous States')
            axs[i_tf].invert_yaxis()

        fig.tight_layout()

    elif form == 'surface':

        fig = plt.figure(figsize=(24, 12))
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        axs = [ax0, ax1]

        for i_tf, trial_filter in enumerate(trial_filters):

            data = dsupr_results[trial_filter]['mean'][train_or_test_id, ...]
            x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

            axs[i_tf].plot_surface(
                x, y, data,
                cmap='viridis',
                alpha=0.7
            )

            axs[i_tf].set_title('DSUP Ratio')
            # axs[i_tf].set_xticklabels(['LDS'] + ns_discrete_states)
            # axs[i_tf].set_yticklabels(ns_states)
            axs[i_tf].set_xlabel('# of Discrete States')
            axs[i_tf].set_ylabel('# of Continuous States')
            axs[i_tf].set_zlim(vmin, vmax)

    elif form == 'lines':

        fig, axs = plt.subplots(1, 2, figsize=(45*mm, 22.5*mm), sharey=True)

        for i_tf, trial_filter in enumerate(trial_filters):
            
            ## Find the simplest model with performance better than best result + its standard error
            performance_reference = utils_vis.model_selection(
                dsupr_results[trial_filter]['mean'][train_or_test_id, ...],
                dsupr_results[trial_filter]['se'][train_or_test_id, ...],
                ns_states, 
                ns_discrete_states,
                includes_pca=False,
                higher_is_better=True)
            
            # axs[i_tf].axhline(
            #     y=performance_reference,
            #     color='black',
            #     lw=size_line_thin,
            #     alpha=alpha_line_thin,
            #     linestyle='--')

            for i_ds, n_discrete_state in reversed(list(enumerate(ns_discrete_states))):

                dsup_mean = dsupr_results[trial_filter]['mean'][train_or_test_id, :, i_ds]
                dsup_se = dsupr_results[trial_filter]['se'][train_or_test_id, :, i_ds]

                # axs[i_tf].plot(
                #     ns_states,
                #     dsupr_results[trial_filter]['mean'][train_or_test_id, :, i_ds],
                #     color=discrete_state_colors[i_ds],
                #     alpha=alpha_line,
                #     lw=0.5)
                
                axs[i_tf].fill_between(
                    ns_states,
                    dsup_mean - dsup_se, 
                    dsup_mean + dsup_se,
                    color=discrete_state_colors[i_ds], 
                    alpha=alpha_line, 
                    linewidth=0)
            
            axs[i_tf].set_xticks(ns_states)

            ## Create tick labels: label only at even indices, else empty string
            # tick_labels = [str(x) if i % 2 == 0 else '' for i, x in enumerate(ns_states)]

            ## Set tick labels
            # axs[i_tf].set_xticklabels(tick_labels)
            axs[i_tf].set_xticklabels([])
            # axs[i_tf].set_yticklabels([])

            # axs[i_tf].set_xlabel('Number of continuous states', fontsize=5)
            # axs[i_tf].set_ylabel('DSUP Ratio', fontsize=5)
            axs[i_tf].set_ylim(vmin, vmax)

            ## Remove top and right spines
            axs[i_tf].spines['top'].set_visible(False)
            axs[i_tf].spines['right'].set_visible(False)

        ## Add custom legend
        # fig.subplots_adjust(bottom=0.2)  ## Increase the bottom margin

        # custom_lines = [
        #     Line2D([0], [0], color=discrete_state_colors[0], lw=2, linestyle='-'),
        #     Line2D([0], [0], color=discrete_state_colors[1], lw=2, linestyle='-'),
        #     Line2D([0], [0], color=discrete_state_colors[2], lw=2, linestyle='-'),
        #     Line2D([0], [0], color=discrete_state_colors[3], lw=2, linestyle='-'),
        #     Line2D([0], [0], color=discrete_state_colors[4], lw=2, linestyle='-'),
        #     Line2D([0], [0], color=discrete_state_colors[5], lw=2, linestyle='-'),
        #     Line2D([0], [0], color=discrete_state_colors[6], lw=2, linestyle='-'),
        #     Line2D([0], [0], color=discrete_state_colors[7], lw=2, linestyle='-')]

        # legend_labels = ['1', '2', '3', '4', '6', '8', '12', '16']

        # ## Create legend with custom elements
        # plt.legend(
        #     custom_lines, 
        #     legend_labels, 
        #     title='# of Discrete States',
        #     loc='upper center', 
        #     bbox_to_anchor=(-0.2, -0.15), 
        #     fancybox=False, 
        #     shadow=False, 
        #     ncol=8)

    fig.tight_layout()

    ## Write image
    session_data_names_str = str(len(session_data_names)) + '_sessions'

    img_name = '_'.join(map(str, [x for x in [
        task_name,
        session_data_names_str,
        'dsupr',
        train_or_test,
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        train_test_option,
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha,
        form] if x is not None]))

    save_path = os.path.join(vis_dir, img_name + '.pdf')

    fig.savefig(save_path, dpi=600, transparent=False, bbox_inches=None, format='pdf')
    plt.close(fig)


def plot_per_time_dsupr_results_avg_session(
    unit_filter,
    input_unit_filter,
    data_format,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    truncate_percentile=10,
    visual_delay_time=0):

    print('Plotting dsupr results...')

    rSLDS_same_speed_dsupr_name = '_'.join(map(str, [x for x in [
        'dsupr_totf',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        'same_speed',
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))
    
    rSLDS_cross_speed_dsupr_name = '_'.join(map(str, [x for x in [
        'dsupr_totf',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        'cross_speed',
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))

    rSLDS_same_speed_dsupr_slow_all = []
    rSLDS_same_speed_dsupr_fast_all = []
    rSLDS_cross_speed_dsupr_slow_all = []
    rSLDS_cross_speed_dsupr_fast_all = []
    trial_length_slow_min_all = []
    trial_length_fast_min_all = []

    for session_data_name in session_data_names:

        session_results_dir = os.path.join(results_dir, session_data_name)

        rSLDS_same_speed_dsupr = np.load(os.path.join(session_results_dir, rSLDS_same_speed_dsupr_name + '.npz'))
        rSLDS_same_speed_dsupr_slow = rSLDS_same_speed_dsupr['dsupr_slow_test_per_time']
        rSLDS_same_speed_dsupr_fast = rSLDS_same_speed_dsupr['dsupr_fast_test_per_time']
        trial_lengths_slow_same_speed = rSLDS_same_speed_dsupr['trial_lengths_slow'] - 1
        trial_lengths_fast_same_speed = rSLDS_same_speed_dsupr['trial_lengths_fast'] - 1

        rSLDS_cross_speed_dsupr = np.load(os.path.join(session_results_dir, rSLDS_cross_speed_dsupr_name + '.npz'))
        rSLDS_cross_speed_dsupr_slow = rSLDS_cross_speed_dsupr['dsupr_slow_test_per_time']
        rSLDS_cross_speed_dsupr_fast = rSLDS_cross_speed_dsupr['dsupr_fast_test_per_time']
        trial_lengths_slow_cross_speed = rSLDS_cross_speed_dsupr['trial_lengths_slow'] - 1
        trial_lengths_fast_cross_speed = rSLDS_cross_speed_dsupr['trial_lengths_fast'] - 1

        assert np.array_equal(trial_lengths_slow_same_speed, trial_lengths_slow_cross_speed)
        assert np.array_equal(trial_lengths_fast_same_speed, trial_lengths_fast_cross_speed)
        trial_lengths_slow = trial_lengths_slow_same_speed
        trial_lengths_fast = trial_lengths_fast_same_speed

        # Take the first discrete state (LDS) and last n_iters
        rSLDS_same_speed_dsupr_slow = rSLDS_same_speed_dsupr_slow[:, :, 0, -1]
        rSLDS_same_speed_dsupr_fast = rSLDS_same_speed_dsupr_fast[:, :, 0, -1]
        rSLDS_cross_speed_dsupr_slow = rSLDS_cross_speed_dsupr_slow[:, :, 0, -1]
        rSLDS_cross_speed_dsupr_fast = rSLDS_cross_speed_dsupr_fast[:, :, 0, -1]

        # Truncate the dsupr results with specified percentile
        rSLDS_same_speed_dsupr_slow, trial_length_slow_min = drop_and_truncate(rSLDS_same_speed_dsupr_slow, trial_lengths_slow, truncate_percentile)
        rSLDS_same_speed_dsupr_fast, trial_length_fast_min = drop_and_truncate(rSLDS_same_speed_dsupr_fast, trial_lengths_fast, truncate_percentile)
        rSLDS_cross_speed_dsupr_slow, _ = drop_and_truncate(rSLDS_cross_speed_dsupr_slow, trial_lengths_slow, truncate_percentile)
        rSLDS_cross_speed_dsupr_fast, _ = drop_and_truncate(rSLDS_cross_speed_dsupr_fast, trial_lengths_fast, truncate_percentile)

        trial_length_slow_min_all.append(trial_length_slow_min)
        trial_length_fast_min_all.append(trial_length_fast_min)

        # Average over random states
        rSLDS_same_speed_dsupr_slow = np.mean(rSLDS_same_speed_dsupr_slow, axis=0)
        rSLDS_same_speed_dsupr_fast = np.mean(rSLDS_same_speed_dsupr_fast, axis=0)
        rSLDS_cross_speed_dsupr_slow = np.mean(rSLDS_cross_speed_dsupr_slow, axis=0)
        rSLDS_cross_speed_dsupr_fast = np.mean(rSLDS_cross_speed_dsupr_fast, axis=0)

        # Append to all sessions
        rSLDS_same_speed_dsupr_slow_all.append(rSLDS_same_speed_dsupr_slow)
        rSLDS_same_speed_dsupr_fast_all.append(rSLDS_same_speed_dsupr_fast)
        rSLDS_cross_speed_dsupr_slow_all.append(rSLDS_cross_speed_dsupr_slow)
        rSLDS_cross_speed_dsupr_fast_all.append(rSLDS_cross_speed_dsupr_fast)

    ## Combine all sessions

    # Get the minimum trial length across all sessions
    trial_length_slow_min = np.min(trial_length_slow_min_all)
    trial_length_fast_min = np.min(trial_length_fast_min_all)

    # Get the number of trials across all sessions
    n_trials_slow_all = np.sum([x.shape[1] for x in rSLDS_same_speed_dsupr_slow_all])
    n_trials_fast_all = np.sum([x.shape[1] for x in rSLDS_same_speed_dsupr_fast_all])

    n_ns_states = len(ns_states)
    rSLDS_same_speed_dsupr_slow_all_ = np.zeros((n_ns_states, n_trials_slow_all, trial_length_slow_min))
    rSLDS_same_speed_dsupr_fast_all_ = np.zeros((n_ns_states, n_trials_fast_all, trial_length_fast_min))
    rSLDS_cross_speed_dsupr_slow_all_ = np.zeros((n_ns_states, n_trials_slow_all, trial_length_slow_min))
    rSLDS_cross_speed_dsupr_fast_all_ = np.zeros((n_ns_states, n_trials_fast_all, trial_length_fast_min))

    n_trials_slow_cumulative = 0
    n_trials_fast_cumulative = 0
    for i in range(len(session_data_names)):
        n_trials_slow = rSLDS_same_speed_dsupr_slow_all[i].shape[1]
        n_trials_fast = rSLDS_same_speed_dsupr_fast_all[i].shape[1]

        # Truncate to the minimum trial length
        rSLDS_same_speed_dsupr_slow_all[i] = rSLDS_same_speed_dsupr_slow_all[i][..., :trial_length_slow_min]
        rSLDS_same_speed_dsupr_fast_all[i] = rSLDS_same_speed_dsupr_fast_all[i][..., :trial_length_fast_min]
        rSLDS_cross_speed_dsupr_slow_all[i] = rSLDS_cross_speed_dsupr_slow_all[i][..., :trial_length_slow_min]
        rSLDS_cross_speed_dsupr_fast_all[i] = rSLDS_cross_speed_dsupr_fast_all[i][..., :trial_length_fast_min]

        # Concatenate all sessions
        rSLDS_same_speed_dsupr_slow_all_[:, n_trials_slow_cumulative:n_trials_slow_cumulative + n_trials_slow, :] = rSLDS_same_speed_dsupr_slow_all[i]
        rSLDS_same_speed_dsupr_fast_all_[:, n_trials_fast_cumulative:n_trials_fast_cumulative + n_trials_fast, :] = rSLDS_same_speed_dsupr_fast_all[i]
        rSLDS_cross_speed_dsupr_slow_all_[:, n_trials_slow_cumulative:n_trials_slow_cumulative + n_trials_slow, :] = rSLDS_cross_speed_dsupr_slow_all[i]
        rSLDS_cross_speed_dsupr_fast_all_[:, n_trials_fast_cumulative:n_trials_fast_cumulative + n_trials_fast, :] = rSLDS_cross_speed_dsupr_fast_all[i]

        n_trials_slow_cumulative += n_trials_slow
        n_trials_fast_cumulative += n_trials_fast

    dsupr_results = {
        # 'fast':
        'near':
            {
                'LDS_mean_same_speed': rSLDS_same_speed_dsupr_fast_all_,
                'LDS_mean_cross_speed': rSLDS_cross_speed_dsupr_fast_all_,
                # 'LDS_mean_same_speed': rSLDS_same_speed_dsupr_fast_all[0],
                # 'LDS_mean_cross_speed': rSLDS_cross_speed_dsupr_fast_all[0],
            },
        # 'slow':
        'far':
            {
                'LDS_mean_same_speed': rSLDS_same_speed_dsupr_slow_all_,
                'LDS_mean_cross_speed': rSLDS_cross_speed_dsupr_slow_all_,
                # 'LDS_mean_same_speed': rSLDS_same_speed_dsupr_slow_all[0],
                # 'LDS_mean_cross_speed': rSLDS_cross_speed_dsupr_slow_all[0],
            }
    }
    
    vmin = 0
    vmax = 1
    
    ## Plot results

    fig1 = plt.figure(figsize=(24, 12))
    ax0 = fig1.add_subplot(1, 2, 1, projection='3d')
    ax1 = fig1.add_subplot(1, 2, 2, projection='3d')
    axs = [ax0, ax1]

    for i_tf, trial_filter in enumerate(trial_filters):

        # Plot lines and fill under them
        for i_cs, n_states in enumerate(ns_states):

            color_list = ['blue', 'red']

            zs_list = [
                np.mean(dsupr_results[trial_filter]['LDS_mean_same_speed'], axis=1)[i_cs],
                np.mean(dsupr_results[trial_filter]['LDS_mean_cross_speed'], axis=1)[i_cs]
            ]

            for color, zs in zip(color_list, zs_list):

                trial_length = zs.shape[-1]

                xs = np.arange(trial_length)
                ys = np.full(trial_length, n_states)

                # Plot the line
                axs[i_tf].plot(
                    xs, 
                    ys, 
                    zs, 
                    color=color, 
                    lw=size_line,
                    alpha=alpha_line)

                # Create the polygon for filling
                verts = utils_vis.polygon_under_curve(xs, ys, zs, z_min=vmin)

                # Create a Poly3DCollection object
                poly = Poly3DCollection(
                    verts, 
                    facecolor=color, 
                    alpha=alpha_line_thin)

                # Add the polygon to the axes
                axs[i_tf].add_collection3d(poly)

        if trial_filter in ['slow', 'far']:
            xtick_step = 5
        else:
            xtick_step = 1

        axs[i_tf].set_xticks(xs[::xtick_step])
        axs[i_tf].set_xticklabels(np.round(xs[::xtick_step] * config.time_step + visual_delay_time, 2))
        axs[i_tf].set_yticks(ns_states)


        # Set labels and title
        axs[i_tf].set_xlabel('Time (s)', fontsize=label_fontsize)
        axs[i_tf].set_ylabel('# of Latent States', fontsize=label_fontsize)
        axs[i_tf].set_zlabel('Absolute Angle Error (degrees)', fontsize=label_fontsize)

        # axs[i_tf].set_xlim(ns_states[0], ns_states[-1])
        # axs[i_tf].set_ylim(ns_discrete_states[0], ns_discrete_states[-1])
        axs[i_tf].set_zlim(vmin, vmax)
        
        # axs[i_tf].set_yticklabels(['PCA'] + ns_discrete_states)

        # Adjust viewing angle
        # axs[i_tf].view_init(elev=10, azim=-75)
        axs[i_tf].view_init(elev=20, azim=-40)

        # axs[i_tf].tick_params(axis='both', which='major', labelsize=label_fontsize)

        ## Remove background panes
        axs[i_tf].xaxis.pane.fill = False
        axs[i_tf].yaxis.pane.fill = False
        axs[i_tf].zaxis.pane.fill = False

        ## Set grid colors
        # axs[i_tf].xaxis._axinfo['grid'].update(color='black')
        # axs[i_tf].yaxis._axinfo['grid'].update(color='black')
        # axs[i_tf].zaxis._axinfo['grid'].update(color='black')    

        # ## Create tick labels: label only at even indices, else empty string
        # tick_labels = [str(x) if i % 2 == 0 else '' for i, x in enumerate(ns_states)]

        # ## Set tick labels
        # axs[i_tf].set_xticklabels(tick_labels)


    ## Add custom legend
    # fig.subplots_adjust(bottom=0.2)  ## Increase the bottom margin

    # custom_lines = [
    #     Line2D([0], [0], color='blue',                   lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[0], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[1], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[2], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[3], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[4], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[5], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[6], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[7], lw=2, linestyle='-')]

    # legend_labels = ['PCA', '1', '2', '3', '4', '6', '8', '12', '16']

    # ## Create legend with custom elements
    # plt.legend(
    #     custom_lines, 
    #     legend_labels, 
    #     title='# of Discrete States',
    #     loc='upper center', 
    #     bbox_to_anchor=(-0.2, -0.15), 
    #     fancybox=False, 
    #     shadow=False, 
    #     ncol=9)

    ## Write image
    if len(session_data_names) > 1:
        session_data_names_str = str(len(session_data_names)) + '_sessions'
    else:
        session_data_names_str = str(session_data_names)

    img_name = '_'.join(map(str, [x for x in [
        session_data_names_str,
        'per_time_dsupr',
        truncate_percentile,
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        dynamics_class,
        emission_class,
        init_type,
        alpha] if x is not None]))

    save_path = os.path.join(vis_dir, img_name + '.pdf')

    fig1.savefig(save_path, dpi=600, transparent=True)
    # plt.show()
    # print('(Axis 0) Elevation angle:', axs[0].elev, ' Azimuth angle:', axs[0].azim)
    # print('(Axis 1) Elevation angle:', axs[1].elev, ' Azimuth angle:', axs[1].azim)
    plt.close(fig1)


def plot_elbos_avg_session(
    unit_filter,
    input_unit_filter,
    data_format,
    train_test_option,
    n_iters,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    train_or_test='test',
    form='heatmap'):

    print('Plotting ELBOs...')

    xticks = np.arange(len(ns_states))
    xticklabels = ns_states

    rSLDS_elbos_name = '_'.join(map(str, [x for x in [
        'elbos',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        # trial_filters,
        ['slow', 'fast'],
        train_test_option,
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        n_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))

    rSLDS_elbos_slow_all = []
    rSLDS_elbos_fast_all = []

    for session_data_name in session_data_names:

        session_results_dir = os.path.join(results_dir, session_data_name)

        rSLDS_elbos = np.load(os.path.join(session_results_dir, rSLDS_elbos_name + '.npz'))
        rSLDS_elbos_slow = rSLDS_elbos['elbos_slow']
        rSLDS_elbos_fast = rSLDS_elbos['elbos_fast']

        ## Take the last n_iters, then add to all sessions
        rSLDS_elbos_slow = rSLDS_elbos_slow[..., -1]
        rSLDS_elbos_fast = rSLDS_elbos_fast[..., -1]

        rSLDS_elbos_slow_all.append(rSLDS_elbos_slow)
        rSLDS_elbos_fast_all.append(rSLDS_elbos_fast)

    ## Stack all sessions
    rSLDS_elbos_slow_all = np.stack(rSLDS_elbos_slow_all,  axis=0)
    rSLDS_elbos_fast_all = np.stack(rSLDS_elbos_fast_all,  axis=0)

    ## Compute mean and standard error over sessions, random states, and folds
    rSLDS_elbos_slow_all_mean = np.mean(rSLDS_elbos_slow_all, axis=(0, 2, 3))
    rSLDS_elbos_fast_all_mean = np.mean(rSLDS_elbos_fast_all, axis=(0, 2, 3))

    correction_factor = np.sqrt(len(session_data_names) * len(random_states) * n_folds)
    rSLDS_elbos_slow_all_se = np.std(rSLDS_elbos_slow_all, axis=(0, 2, 3)) / correction_factor
    rSLDS_elbos_fast_all_se = np.std(rSLDS_elbos_fast_all, axis=(0, 2, 3)) / correction_factor


    elbos_results = {
        'slow':
            {
                'mean': rSLDS_elbos_slow_all_mean,
                'se': rSLDS_elbos_slow_all_se,
            },
        'fast':
            {
                'mean': rSLDS_elbos_fast_all_mean,
                'se': rSLDS_elbos_fast_all_se,
            }
    }

    if train_or_test == 'train':
        train_or_test_id = 0
    elif train_or_test == 'test':
        train_or_test_id = 1

    vmin_all = np.min([np.min(elbos_results['slow']['mean'][train_or_test_id]), np.min(elbos_results['fast']['mean'][train_or_test_id])])
    vmax_all = np.max([np.max(elbos_results['slow']['mean'][train_or_test_id]), np.max(elbos_results['fast']['mean'][train_or_test_id])])

    vmin = vmin_all - 10
    vmax = vmax_all + 10


    ## Plot results
    if form == 'heatmap':

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        for i_tf, trial_filter in enumerate(trial_filters):

            sns.heatmap(
                elbos_results[trial_filter]['mean'][train_or_test_id, ...], 
                ax=axs[i_tf], 
                cbar=True,
                # vmin=vmin,
                # vmax=vmax,
                cmap='viridis')

            axs[i_tf].set_title('ELBOs')
            # axs[i_tf].set_xticklabels(['LDS'] + ns_discrete_states)
            axs[i_tf].set_xticklabels(ns_discrete_states)
            axs[i_tf].set_yticklabels(ns_states)
            axs[i_tf].set_xlabel('# of Discrete States')
            axs[i_tf].set_ylabel('# of Continuous States')
            axs[i_tf].invert_yaxis()

        fig.tight_layout()


    elif form == 'surface':

        fig = plt.figure(figsize=(24, 12))
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        axs = [ax0, ax1]

        for i_tf, trial_filter in enumerate(trial_filters):

            data = elbos_results[trial_filter]['mean'][train_or_test_id, ...]
            x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

            axs[i_tf].plot_surface(
                x, y, data,
                cmap='viridis',
                alpha=0.7
            )

            axs[i_tf].set_title('ELBOs')
            # axs[i_tf].set_xticklabels(['LDS'] + ns_discrete_states)
            # axs[i_tf].set_yticklabels(ns_states)
            axs[i_tf].set_xlabel('# of Discrete States')
            axs[i_tf].set_ylabel('# of Continuous States')
            # axs[i_tf].set_zlim(vmin, vmax)


    elif form == 'waterfall':

        fig = plt.figure(figsize=(24, 12))
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        axs = [ax0, ax1]

        for i_tf, trial_filter in enumerate(trial_filters):

            # Find the simplest model with performance better than best result + its standard error
            performance_reference = utils_vis.model_selection(
                elbos_results[trial_filter]['mean'][train_or_test_id, ...],
                elbos_results[trial_filter]['se'][train_or_test_id, ...],
                ns_states,
                ns_discrete_states,
                includes_pca=False,
                higher_is_better=True)

            # Plot lines and fill under them
            for i_cs in range(len(ns_states)):

                xs = np.arange(len(ns_discrete_states))
                ys = np.full_like(ns_discrete_states, i_cs)
                zs = elbos_results[trial_filter]['mean'][train_or_test_id, i_cs]

                # Plot the line
                axs[i_tf].plot(
                    xs, 
                    ys, 
                    zs, 
                    color=continuous_state_colors[i_cs], 
                    lw=size_line,
                    alpha=alpha_line)

                # Create the polygon for filling
                verts = utils_vis.polygon_under_curve(xs, ys, zs, z_min=vmin)

                # Create a Poly3DCollection object
                poly = Poly3DCollection(
                    verts, 
                    facecolor=continuous_state_colors[i_cs], 
                    alpha=alpha_line_thin)

                # Add the polygon to the axes
                axs[i_tf].add_collection3d(poly)

            # Set labels and title
            axs[i_tf].set_ylabel('# of Continuous States', fontsize=label_fontsize)
            axs[i_tf].set_xlabel('# of Discrete States', fontsize=label_fontsize)
            axs[i_tf].set_zlabel('ELBOs', fontsize=label_fontsize)

            # axs[i_tf].set_xlim(ns_states[0], ns_states[-1])
            # axs[i_tf].set_ylim(ns_discrete_states[0], ns_discrete_states[-1])
            axs[i_tf].set_zlim(vmin, vmax)

            axs[i_tf].set_yticks(np.arange(len(ns_states)))
            axs[i_tf].set_yticklabels(ns_states)
            axs[i_tf].set_xticks(np.arange(len(ns_discrete_states)))
            axs[i_tf].set_xticklabels(ns_discrete_states)

            # Adjust viewing angle
            axs[i_tf].view_init(elev=20, azim=-40)

            # axs[i_tf].tick_params(axis='both', which='major', labelsize=label_fontsize)

            ## Remove background panes
            axs[i_tf].xaxis.pane.fill = False
            axs[i_tf].yaxis.pane.fill = False
            axs[i_tf].zaxis.pane.fill = False

            ## Set grid colors
            # axs[i_tf].xaxis._axinfo['grid'].update(color='black')
            # axs[i_tf].yaxis._axinfo['grid'].update(color='black')
            # axs[i_tf].zaxis._axinfo['grid'].update(color='black')    

            # ## Create tick labels: label only at even indices, else empty string
            tick_labels = [str(x) if i % 2 == 0 else '' for i, x in enumerate(ns_states)]

            # ## Set tick labels
            axs[i_tf].set_yticklabels(tick_labels)

            ## Visualize the reference plane
            # Create 2D meshgrid for x (discrete states) and y (continuous states)
            X, Y = np.meshgrid(
                np.arange(len(ns_discrete_states)),  # x-coordinates
                np.arange(len(ns_states))            # y-coordinates
            )

            # Z will be filled with the reference-performance value
            Z = np.full_like(X, performance_reference, dtype=float)
            # ipdb.set_trace()
            print(performance_reference)

            axs[i_tf].plot_surface(
                X,      # x-coordinates
                Y,      # y-coordinates
                Z,      # z-coordinates
                color='gray',
                alpha=0.3,
                shade=False
            )

        
    ## Write image
    if len(session_data_names) > 3:
        session_data_names_str = str(len(session_data_names)) + '_sessions'
    else:
        session_data_names_str = str(session_data_names)

    img_name = '_'.join(map(str, [x for x in [
        session_data_names_str,
        'elbos',
        train_or_test,
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        train_test_option,
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha,
        form] if x is not None]))

    save_path = os.path.join(vis_dir, img_name + '.pdf')

    fig.savefig(save_path, dpi=600, transparent=False)
    # plt.show()
    print('(Axis 0) Elevation angle:', axs[0].elev, ' Azimuth angle:', axs[0].azim)
    print('(Axis 1) Elevation angle:', axs[1].elev, ' Azimuth angle:', axs[1].azim)
    plt.close(fig)


def plot_entropy_results_avg_session(
    unit_filter,
    input_unit_filter,
    data_format,
    train_test_option,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    train_or_test='test',
    form='heatmap'):

    print('Plotting Entropy Difference results...')

    task_name = session_data_names[0].split('_')[-1]

    xticks = np.arange(len(ns_states))
    xticklabels = ns_states

    rSLDS_entropy_name = '_'.join(map(str, [x for x in [
        'entropy',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        train_test_option,
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))

    rSLDS_entropy_slow_all = []
    rSLDS_entropy_fast_all = []

    for session_data_name in session_data_names:

        session_results_dir = os.path.join(results_dir, session_data_name)

        rSLDS_entropy = np.load(os.path.join(session_results_dir, rSLDS_entropy_name + '.npz'))

        rSLDS_entropy_slow = rSLDS_entropy['entropy_per_trial_slow'] - rSLDS_entropy['entropy_per_time_slow']
        rSLDS_entropy_fast = rSLDS_entropy['entropy_per_trial_fast'] - rSLDS_entropy['entropy_per_time_fast']

        ## Take the last n_iters, then add to all sessions
        rSLDS_entropy_slow = rSLDS_entropy_slow[..., -1]
        rSLDS_entropy_fast = rSLDS_entropy_fast[..., -1]

        rSLDS_entropy_slow_all.append(rSLDS_entropy_slow)
        rSLDS_entropy_fast_all.append(rSLDS_entropy_fast)

    ## Stack all sessions
    rSLDS_entropy_slow_all = np.stack(rSLDS_entropy_slow_all, axis=0)
    rSLDS_entropy_fast_all = np.stack(rSLDS_entropy_fast_all, axis=0)

    ## Compute mean and standard error over sessions, random states, and folds
    rSLDS_entropy_slow_all_mean = np.mean(rSLDS_entropy_slow_all, axis=(0, 2, 3))
    rSLDS_entropy_fast_all_mean = np.mean(rSLDS_entropy_fast_all, axis=(0, 2, 3))

    correction_factor = np.sqrt(len(session_data_names) * len(random_states) * n_folds)
    rSLDS_entropy_slow_all_se = np.std(rSLDS_entropy_slow_all, axis=(0, 2, 3)) / correction_factor
    rSLDS_entropy_fast_all_se = np.std(rSLDS_entropy_fast_all, axis=(0, 2, 3)) / correction_factor

    entropy_results = {
        # 'slow':
        'far':
            {
                'mean': rSLDS_entropy_slow_all_mean,
                'se': rSLDS_entropy_slow_all_se,
            },
        # 'fast':
        'near':
            {
                'mean': rSLDS_entropy_fast_all_mean,
                'se': rSLDS_entropy_fast_all_se,
            }
    }

    vmin = -0.5
    vmax = 0.5

    if train_or_test == 'train':
        train_or_test_id = 0
    elif train_or_test == 'test':
        train_or_test_id = 1

    ## Plot results
    if form == 'heatmap':

        fig, axs = plt.subplots(1, 2, figsize=(45*mm, 22.5*mm))

        for i_tf, trial_filter in enumerate(trial_filters):

            # Find the simplest model with performance better than best result + its standard error
            performance_reference = utils_vis.model_selection(
                entropy_results[trial_filter]['mean'][train_or_test_id, ...],
                entropy_results[trial_filter]['se'][train_or_test_id, ...],
                ns_states,
                ns_discrete_states,
                includes_pca=False,
                higher_is_better=True)
            print(performance_reference)
            
            sns.heatmap(
                entropy_results[trial_filter]['mean'][train_or_test_id, :, 1:].T, # Exclude LDS
                ax=axs[i_tf],
                vmin=vmin,
                vmax=vmax,
                cmap='mako',
                cbar=False,          # ← suppress the color-bar
                # square=True,         # ← make the axes a true square
                xticklabels=False,   # ← drop x-axis tick labels
                yticklabels=False,   # ← drop y-axis tick labels
                linewidths=0,
                linecolor=None,
                rasterized=True,
            )
        
            # If you also want to hide the tick marks themselves:
            axs[i_tf].tick_params(left=False, bottom=False)
            axs[i_tf].set_box_aspect(1)

            # axs[i_tf].set_title('Entropy Difference')
            # axs[i_tf].set_xticklabels(['LDS'] + ns_discrete_states)
            # axs[i_tf].set_yticklabels(ns_discrete_states[1:])
            # axs[i_tf].set_xticklabels(ns_states)
            # axs[i_tf].set_ylabel('Number of discrete states', fontsize=5)
            # axs[i_tf].set_xlabel('Number of continuous states', fontsize=5)
            axs[i_tf].invert_yaxis()

            # axs[i_tf].set_aspect('equal')

        fig.tight_layout()


    elif form == 'surface':

        fig = plt.figure(figsize=(24, 12))
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        axs = [ax0, ax1]

        for i_tf, trial_filter in enumerate(trial_filters):

            data = entropy_results[trial_filter]['mean'][train_or_test_id, ...]
            x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

            axs[i_tf].plot_surface(
                x, y, data,
                cmap='viridis',
                alpha=0.7
            )

            axs[i_tf].set_title('Entropy Difference')
            # axs[i_tf].set_xticklabels(['LDS'] + ns_discrete_states)
            # axs[i_tf].set_yticklabels(ns_states)
            axs[i_tf].set_xlabel('# of Discrete States')
            axs[i_tf].set_ylabel('# of Continuous States')
            axs[i_tf].set_zlim(vmin, vmax)


    ## Write image
    session_data_names_str = str(len(session_data_names)) + '_sessions'

    img_name = '_'.join(map(str, [x for x in [
        task_name,
        session_data_names_str,
        'entropy',
        train_or_test,
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        train_test_option,
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha,
        form] if x is not None]))

    save_path = os.path.join(vis_dir, img_name + '.pdf')

    fig.savefig(save_path, dpi=600, transparent=True, bbox_inches=None, format='pdf')
    plt.close(fig)


    # Create a continuous colormap from seaborn's 'rocket' palette
    cmap = sns.color_palette("mako", as_cmap=True)
    norm = Normalize(vmin=2, vmax=20)

    # Build the figure
    fig, ax = plt.subplots(figsize=(20*mm, 2*mm))
    # fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Add a horizontal colorbar without ticks or labels
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='horizontal'
    )
    cbar.set_ticks([])
    cbar.outline.set_visible(False)

    # Remove the axes frame completely
    ax.set_frame_on(False)
    fig.savefig(os.path.join(vis_dir, 'entropy_color_bar.pdf'), bbox_inches=None, transparent=True, dpi=600, format='pdf')




if __name__ == '__main__':
    
    # for (
    #     unit_filter,
    #     input_unit_filter,
    #     data_format,
    #     label_format,
    #     dynamics_class,
    #     emission_class,
    #     init_type,
    #     subspace_type,
    #     alpha) in itertools.product(
    #         unit_filters,
    #         input_unit_filters,
    #         data_formats,
    #         label_formats,
    #         dynamics_classes,
    #         emission_classes,
    #         init_types,
    #         subspace_types,
    #         alphas):
        
    #     plot_per_time_decoding_results_avg_session(
    #         unit_filter,
    #         input_unit_filter,
    #         data_format,
    #         label_format,
    #         dynamics_class,
    #         emission_class,
    #         init_type,
    #         subspace_type,
    #         alpha,
    #         truncate_percentile=10,
    #         visual_delay_time=0.1)
    
    for (
        unit_filter,
        input_unit_filter,
        data_format,
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha) in itertools.product(
            unit_filters,
            input_unit_filters,
            data_formats,
            dynamics_classes,
            emission_classes,
            init_types,
            subspace_types,
            alphas):
        
        plot_per_time_inference_results_avg_session(
            unit_filter,
            input_unit_filter,
            data_format,
            dynamics_class,
            emission_class,
            init_type,
            subspace_type,
            alpha,
            inference_type='forecast',
            truncate_percentile=10,
            visual_delay_time=0.275)
        
        # plot_per_time_dsupr_results_avg_session(
        #     unit_filter,
        #     input_unit_filter,
        #     data_format,
        #     dynamics_class,
        #     emission_class,
        #     init_type,
        #     subspace_type,
        #     alpha,
        #     truncate_percentile=10,
        #     visual_delay_time=0.25)
    
    # for (
    #     unit_filter,
    #     input_unit_filter,
    #     data_format,
    #     label_format,
    #     train_test_option,
    #     dynamics_class,
    #     emission_class,
    #     init_type,
    #     subspace_type,
    #     alpha) in itertools.product(
    #         unit_filters,
    #         input_unit_filters,
    #         data_formats,
    #         label_formats,
    #         train_test_options,
    #         dynamics_classes,
    #         emission_classes,
    #         init_types,
    #         subspace_types,
    #         alphas):
        
    #     plot_decoding_results_avg_session(
    #         unit_filter,
    #         input_unit_filter,
    #         data_format,
    #         label_format,
    #         train_test_option,
    #         dynamics_class,
    #         emission_class,
    #         init_type,
    #         subspace_type,
    #         alpha,
    #         train_or_test='test',
    #         form='scatter')

    for (
        unit_filter,
        input_unit_filter,
        data_format,
        train_test_option,
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha) in itertools.product(
            unit_filters,
            input_unit_filters,
            data_formats,
            train_test_options,
            dynamics_classes,
            emission_classes,
            init_types,
            subspace_types,
            alphas):

        plot_inference_results_avg_session(
            unit_filter,
            input_unit_filter,
            data_format,
            train_test_option,
            dynamics_class,
            emission_class,
            init_type,
            subspace_type,
            alpha,
            train_or_test='test',
            inference_type='forecast',
            form='heatmap')
        
        # plot_dsupr_results_avg_session(
        #     unit_filter,
        #     input_unit_filter,
        #     data_format,
        #     train_test_option,
        #     dynamics_class,
        #     emission_class,
        #     init_type,
        #     subspace_type,
        #     alpha,
        #     train_or_test='test',
        #     form='lines')
        
    # #     # plot_elbos_avg_session(
    # #     #     unit_filter,
    # #     #     input_unit_filter,
    # #     #     data_format,
    # #     #     train_test_option,
    # #     #     25,
    # #     #     dynamics_class,
    # #     #     emission_class,
    # #     #     init_type,
    # #     #     subspace_type,
    # #     #     alpha,
    # #     #     train_or_test='test',
    # #     #     form='waterfall')

    #     plot_entropy_results_avg_session(
    #         unit_filter,
    #         input_unit_filter,
    #         data_format,
    #         train_test_option,
    #         dynamics_class,
    #         emission_class,
    #         init_type,
    #         subspace_type,
    #         alpha,
    #         train_or_test='test',
    #         form='heatmap')
